# 当量比推导过程

已知燃料的化学成分为：

$$96.57\%\mathrm{CH_4}, 0.28\%\mathrm{C_2H_6}, 0.03\%\mathrm{C_3H_8}, 2.03\%\mathrm{CO_2}, 1.06\%\mathrm{N_2}, 0.03\%\mathrm{He}$$

设燃料为$1\mathrm{mol}$，空气为$x\mathrm{mol}$。

气体化学反应：
$$0.9657\quad CH_4+2O_2 \rightarrow CO_2+2H_2O$$
$$0.0028\quad C_2H_6+3.5O_2 \rightarrow 2CO_2+3H_2O$$
$$0.0003\quad C_3H_8+5O_2 \rightarrow 3CO_2+4H_2O$$
$$0.0203\quad CO_2\rightarrow CO_2$$
$$0.0106\quad N_2\rightarrow N_2$$
$$0.0003\quad He\rightarrow He$$
$$0.21x\quad O_2\rightarrow (0.21x-3.8854)O_2$$
$$0.79x\quad N_2\rightarrow (0.79x+0.0106)N_2$$

烃类化学反应方程式：
$$C_{0.9722}H_{3.882}+1.9427O_2\rightarrow0.9722CO_2+1.941H_2O$$

每$1\mathrm{mol}$燃料带来$0.9925\mathrm{mol}$ CO2，消耗$1.9427\mathrm{mol}$ O2，剩余$0.21x-1.9427\mathrm{mol}$ O2。

已知排放烟气中的氧气的摩尔浓度浓度，记为$p$，则

$$\begin{aligned}
p
&=\frac{0.21x-1.9427}{0.9925+(0.0106+0.79x)+0.0003+(0.21x-1.9427)}\\
&=\frac{0.21x-1.9427}{x-0.9393}
\end{aligned}$$

可以反解出

$$x=\frac{1.9427-0.9393p}{0.21-p}$$

刚好完全反应的理论空气量为$1.9427/0.21=9.25095\mathrm{mol}$，而实际空气量为$x$，故当量比为

$$\phi=\frac{9.25095}{x}=\frac{0.21-p}{0.21-0.1015355p}$$












