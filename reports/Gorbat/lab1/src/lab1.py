import numpy as np

C0, C1 = -9, -8
X = np.array([[-9, -9], [-9, -8], [-8, -9], [-8, -8]], float)
Y = np.array([-9, -8, -8, -9], float)
T = (Y - C0) / (C1 - C0)
EE, MAX_EPOCHS, LR = 0.01, 10000, 2.0

sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
dsig = lambda y: y * (1 - y)
norm = lambda x: (np.asarray(x, float) - C0) / (C1 - C0)
scale = lambda y: C0 + y * (C1 - C0)
cls = lambda v: C1 if abs(v - C1) < abs(v - C0) else C0


class MLP:
    def __init__(self):
        s = 0.5
        self.W1, self.b1 = np.random.randn(2, 2) * s, np.zeros(2)
        self.W2, self.b2 = np.random.randn(2, 1) * s, np.zeros(1)

    def fwd(self, x):
        x = norm(x).ravel()
        self.h = sig(x @ self.W1 + self.b1)
        self.o = sig(self.h @ self.W2 + self.b2)
        return self.o.item()

    def step(self, x, t):
        o = self.fwd(x)
        x = norm(x).ravel()
        d = (o - t) * o * (1 - o)
        self.W2 -= LR * self.h.reshape(-1, 1) * d
        self.b2 -= LR * d
        dh = (self.W2.ravel() * d) * dsig(self.h)
        self.W1 -= LR * np.outer(x, dh)
        self.b1 -= LR * dh


class Perceptron:
    def __init__(self):
        self.W, self.b = np.random.randn(2, 1) * 0.5, np.zeros(1)

    def fwd(self, x):
        x = norm(x).ravel()
        self.o = sig(x @ self.W + self.b)
        return self.o.item()

    def step(self, x, t):
        o = self.fwd(x)
        x = norm(x).ravel()
        d = (o - t) * self.o.item() * (1 - self.o.item())
        self.W -= LR * x.reshape(-1, 1) * d
        self.b -= LR * d


def train(net):
    for ep in range(1, MAX_EPOCHS + 1):
        for x, t in zip(X, T):
            net.step(x, t)
        err = sum((net.fwd(x) - t) ** 2 for x, t in zip(X, T))
        if err <= EE:
            return ep, err
    return MAX_EPOCHS, err


def report(name, net, ep, err):
    acc = sum(cls(scale(net.fwd(x))) == y for x, y in zip(X, Y)) / len(Y)
    print(f"\n{name}: эпох={ep}, E={err:.6f}, точность={acc * 100:.0f}%")
    for x, y in zip(X, Y):
        p = scale(net.fwd(x))
        print(f"  ({x[0]:g},{x[1]:g}) -> {p:.4f} ({cls(p):g}), ожид. {y:g}")


def main():
    np.random.seed(0)
    mlp, perc = MLP(), Perceptron()
    me, me_err = train(mlp)
    pe, pe_err = train(perc)
    print("=== Вариант 3 (c0=-9, c1=-8) ===")
    report("MLP 2-2-1", mlp, me, me_err)
    report("Перцептрон 2-1", perc, pe, pe_err)
    print("\n--- Режим ввода (выход: q) ---")
    while True:
        s = input("A B [-10..10]: ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            a, b = map(float, s.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                raise ValueError
            p = scale(mlp.fwd([a, b]))
            print(f"y={p:.4f}, ближе к c={cls(p):g}")
        except ValueError:
            print("Введите два числа из [-10, 10]")


if __name__ == "__main__":
    main()
