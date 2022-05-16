from lagrange import lagrange_polynomial, reorder_points_near
import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def fn(x):
        return np.tanh(10 * x)

    Hs = (0.25, 0.125, 0.0625)
    xx = 0.35

    linearX = [
        [0.25, 0.5],
        [0.25, 0.375],
        [0.3125, 0.375],
    ]
    cubicX = [
        [0.0, 0.25, 0.5, 0.75],
        [0.125, 0.25, 0.375, 0.5],
        [0.25, 0.3125, 0.375, 0.4375],
    ]

    # create plot

    # grid points for linear interpolation
    linearh = []
    linear_err = []

    for h, x in zip(Hs, linearX):
        x = np.array(x)
        x = x[reorder_points_near(xx, x)]
        # linear interpolation
        y = lagrange_polynomial(xx, x, fn(x), n=1)
        relerr = np.abs(y - fn(xx)) / fn(xx)
        linearh.append(h)
        linear_err.append(relerr)

    cubich = []
    cubic_err = []

    # grid points for cubic interpolation
    for h, x in zip(Hs, cubicX):
        x = np.array(x)
        x = x[reorder_points_near(xx, x)]
        # cubic interpolation
        y = lagrange_polynomial(xx, x, fn(x), n=3)
        relerr = np.abs(y - fn(xx)) / fn(xx)
        cubich.append(h)
        cubic_err.append(relerr)

    # plot to see overshoot
    # iterate over this to get data points
    hs, ys = [], []
    h0, dh = 0.25, 0.0625
    xx = 0.35
    # get data points with step size dh
    for i in range(0, 5):
        h = h0 + i * dh
        hs.append(h)
        ys.append(fn(h))
    # reorder points
    hs, ys = np.array(hs), np.array(ys)
    reorder = reorder_points_near(xx, hs)
    # get smooth evaluation points for plotting
    hs_smooth = np.linspace(hs.min() - 0.5, hs.max() + 0.5, 500)
    ys_smooth = np.empty_like(hs_smooth)
    print(hs, ys)
    for i, h in enumerate(hs_smooth):
        # evaluate lagrange poly for each smooth point
        ys_smooth[i] = lagrange_polynomial(
            h,
            hs[reorder],
            ys[reorder],
            n=ys.shape[0] - 1,
        )

    # Error plot
    fig1 = plt.figure(figsize=(5, 3), tight_layout=True)
    ax0 = fig1.add_subplot(111)
    ax0.scatter(linearh, linear_err, label="linear")
    for i, (linear_v, cubic_v) in enumerate(zip(linear_err, cubic_err)):
        ax0.text(linearh[i], linear_err[i], f"{round(linear_v*100, 3)}%")
        ax0.text(cubich[i], cubic_err[i], f"{round(cubic_v*100, 3)}%")
    ax0.scatter(cubich, cubic_err, marker="+", label="cubic")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("h")
    ax0.set_ylabel("relative error")
    ax0.legend()

    # overshoot plot
    fig2 = plt.figure(figsize=(5, 3), tight_layout=True)
    ax1 = fig2.add_subplot(111)
    ax1.plot(hs_smooth, fn(hs_smooth), linestyle=":", label="True")
    ax1.plot(hs_smooth, ys_smooth, label="Lagrange Polynomial")
    ax1.scatter(hs, ys, label="Evaluation Points")
    ax1.set_xlim(hs.min() - 0.05, hs.max() + 0.05)
    ax1.set_ylim(fn(ys).min() - 0.05, fn(ys).max() + 0.05)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    fig1.savefig("plots/lagrange_error.png", dpi=300)
    fig2.savefig("plots/overshoot.png", dpi=300)
    plt.show()
