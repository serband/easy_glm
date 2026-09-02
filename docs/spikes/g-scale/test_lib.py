import time, numpy as np, polars as pl, tabmat as tm
import spike_lib as L
from easy_glm.core.design import DesignSpec
t=time.time(); df=L.make_data(200_000); print('gen', round(time.time()-t,2), df.shape, df['claims'].sum()/df['exposure'].sum())
spec=DesignSpec.from_data(df, L.PREDICTORS, n_bins=20, weight_col='exposure'); print(spec, spec.n_features)
X=spec.build(df); codes=L.var_codes(spec, df)
Xc=L.build_dense_from_codes(spec, codes); print('codes-dense == spec.build:', np.array_equal(X,Xc))
S=L.build_split(spec, codes); print('split == dense:', np.array_equal(S.toarray(), X))
SM=L.build_split_stepmatrix(spec, codes); print('stepmat split == dense:', np.array_equal(SM.toarray(), X), [type(m).__name__ for m in SM.matrices])
rng=np.random.default_rng(1); beta=rng.normal(size=spec.n_features); d=rng.random(len(df)); v=rng.normal(size=len(df))
print('matvec err', np.abs(SM.matvec(beta)-X@beta).max())
print('t_matvec err', np.abs(SM.transpose_matvec(v)-X.T@v).max())
rows=np.sort(rng.choice(len(df), 50000, replace=False)); cols=np.sort(rng.choice(spec.n_features, 60, replace=False))
print('t_matvec rows/cols err', np.abs(SM.transpose_matvec(v, rows, cols)-X[rows][:,cols].T@v[rows]).max())
Sd=X.T@(X*d[:,None]); print('sandwich err', np.abs(SM.sandwich(d)-Sd).max(), 'dense-split', np.abs(S.sandwich(d)-Sd).max())
print('sandwich rows/cols err', np.abs(SM.sandwich(d, rows, cols)-(X[rows][:,cols].T@(X[rows][:,cols]*d[rows,None]))).max())
print('matvec cols err', np.abs(SM.matvec(beta, cols)-X[:,cols]@beta[cols]).max())
w=np.ones(len(df))/len(df); st,m,sd=SM.standardize(w, True, True); st2,m2,sd2=tm.DenseMatrix(X).standardize(w,True,True)
print('std means/stds err', np.abs(m-m2).max(), np.abs(sd-sd2).max())
sub=SM[rows,:]; print('row subset', sub.shape, np.array_equal(sub.toarray(), X[rows]))
# timing of the three ops
for name,M in [('dense',tm.DenseMatrix(X)),('split',S),('stepmat',SM)]:
    t=time.time(); [M.sandwich(d) for _ in range(3)]; ts=(time.time()-t)/3
    t=time.time(); [M.matvec(beta) for _ in range(3)]; tm_=(time.time()-t)/3
    t=time.time(); [M.transpose_matvec(v) for _ in range(3)]; tt=(time.time()-t)/3
    print(f'{name:8s} sandwich {ts*1e3:7.1f} ms  matvec {tm_*1e3:6.1f} ms  t_matvec {tt*1e3:6.1f} ms')
# aggregation
y=df['claims'].to_numpy()/df['exposure'].to_numpy(); w=df['exposure'].to_numpy()
ca,yb,W,g=L.aggregate(spec,codes,y,w); print('agg groups', len(W), 'ratio', len(df)/len(W), 'W sum ok', np.isclose(W.sum(), w.sum()), 'wy ok', np.isclose((W*yb).sum(), (w*y).sum()))
print('groups consistent', all(np.array_equal(ca[k][g], codes[k]) for k in codes))
