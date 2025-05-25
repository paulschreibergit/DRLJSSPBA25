# -----------------------------------------------------------------------------
# Actor-only PPO für JSSP – 100 % eigenständig lauffähig
# -----------------------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch_scatter import scatter_softmax

# ────────────────────────── Hilfsfunktion ──────────────────────────
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, batch_vec):
    """Softmax pro Teilgraph; maskierte Logits → -inf."""
    masked = logits.masked_fill(~mask, float("-inf"))
    return scatter_softmax(masked, batch_vec)  # stable pro Graph

# ─────────────────────────────  Policy  ────────────────────────────
class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GCNConv(last, hidden))
            last = hidden

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x


class PolicyNetwork(nn.Module):
    """Gibt nur eine Aktionsverteilung zurück (kein Critic)."""

    def __init__(self, node_feat_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, hidden)
        self.actor_head = nn.Linear(hidden, 1)  # 1 Logit / Knoten

    def forward(self, data: Batch, mask: torch.Tensor):
        x = self.encoder(data.x, data.edge_index)
        logits = self.actor_head(x).squeeze(-1)                # [N]
        probs = masked_softmax(logits, mask, data.batch)       # [N]
        return Categorical(probs)

# ──────────────────────────  Rollout-Buffer  ───────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, obs, mask, action, logp, reward, done):
        self.obss.append(obs);  self.masks.append(mask)
        self.acts.append(action);self.logps.append(logp)
        self.rews.append(reward);self.dones.append(done)

    def clear(self):
        self.obss, self.masks = [], []
        self.acts, self.logps = [], []
        self.rews, self.dones = [], []

    def compute_returns_adv(self, gamma: float):
        ret, G = [], 0.0
        for r, d in zip(reversed(self.rews), reversed(self.dones)):
            G = r + gamma * G * (1. - float(d))
            ret.insert(0, G)
        returns = torch.tensor(ret, dtype=torch.float32)
        adv = (returns - returns.mean()) / (returns.std() + 1e-5)
        self.returns = returns
        self.advs    = adv
        self.acts    = torch.tensor(self.acts, dtype=torch.long)
        self.logps   = torch.tensor(self.logps, dtype=torch.float32)

# ───────────────────────────── PPO-Trainer ─────────────────────────
class PPOTrainer:
    def __init__(self, policy, lr=3e-4, clip=0.2,
                 epochs=4, batch_size=64, gamma=0.99, ent_coef=0.01):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip, self.epochs = clip, epochs
        self.batch = batch_size; self.gamma = gamma
        self.ent_coef = ent_coef
        self.buf = RolloutBuffer()

    def store(self, *args):
        self.buf.add(*args)

    def update(self):
        if not self.buf.rews: return
        self.buf.compute_returns_adv(self.gamma)
        data = list(zip(self.buf.obss, self.buf.masks, self.buf.acts,
                        self.buf.logps, self.buf.returns, self.buf.advs))

        for _ in range(self.epochs):
            for i in range(0, len(data), self.batch):
                o, m, a, lp_old, ret, adv = zip(*data[i:i+self.batch])
                batch   = Batch.from_data_list(list(o))
                mask_b  = torch.cat(m); a = torch.stack(a)
                lp_old  = torch.stack(lp_old); adv = torch.stack(adv)

                dist = self.policy(batch, mask_b)
                lp   = dist.log_prob(a)
                ratio= (lp - lp_old).exp()
                s1   = ratio * adv
                s2   = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
                act_loss = -torch.min(s1, s2).mean()
                ent = dist.entropy().mean()
                loss = act_loss - self.ent_coef * ent

                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                self.opt.step()
                self.last_entropy = ent.detach()
        self.buf.clear()
        return {"entropy": self.last_entropy.item()}
