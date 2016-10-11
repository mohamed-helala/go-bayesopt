package baysopt

import (
	"fmt"
	"math"
	// "github.com/gonum/blas"
	// "github.com/gonum/blas/blas64"
	// "github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
)

type GpAquisition struct {
	x_max  []float64
	fx_max float64
	gp     *Gp
}

type GpOptimizer struct {
	gp *Gp
}

type BaseData struct {
	X      *mat64.Dense
	D      []interface{}
	Y      *mat64.Vector
	K      *mat64.Dense
	Ky     *mat64.Dense
	nc, dc int
}

type Gp struct {
	X   *mat64.Dense
	Y   *mat64.Vector
	K   *mat64.Dense
	Ky  *mat64.Dense
	P   *mat64.Vector
	opt *GpOptimizer
	acq *GpAquisition
}

type DynGp struct {
	Gp
	bdata *BaseData
	best  int
	mark1 int
	mark2 int
}

func NewDynGp(n, d int) *DynGp {
	gp := &DynGp{bdata: &BaseData{nc: 0, dc: d}}
	gp.opt = &GpOptimizer{&gp.Gp}
	gp.acq = &GpAquisition{gp: &gp.Gp}
	gp.bdata.X = mat64.NewDense(n, d, nil)
	gp.bdata.Y = mat64.NewVector(n, nil)
	gp.K = mat64.NewDense(n, n, nil)
	gp.Ky = mat64.NewDense(n, n, nil)
	gp.bdata.D = make([]interface{}, n)
	gp.P = mat64.NewVector(d+2, nil)
	gp.bdata.nc, gp.bdata.dc = 0, d
	gp.X = &mat64.Dense{}
	gp.Y = &mat64.Vector{}
	gp.best = -1
	gp.mark1 = -1
	gp.mark2 = -1
	return gp
}

func NewGp(x []float64, fx []float64) *Gp {
	n, d := len(fx), len(x)/len(fx)
	X := mat64.NewDense(n, d, x)
	Y := mat64.NewVector(n, fx)
	K := mat64.NewDense(n, n, nil)
	Ky := mat64.NewDense(n, n, nil)
	P := mat64.NewVector(d+2, nil)
	gp := &Gp{X: X, Y: Y, K: K, Ky: Ky, P: P}
	gp.opt = &GpOptimizer{gp}
	gp.acq = &GpAquisition{gp: gp}
	return gp
}

func (gp *Gp) Cov(_P []float64) {

	n, d := gp.X.Dims()
	P := mat64.NewVector(d+2, _P)
	if gp.P.Equals(P) {
		return
	}
	T := mat64.NewVector(d, nil)
	L := mat64.NewVector(d, nil)
	LL := P.ViewVec(0, d)
	L.Exp(LL)
	sigma_f := math.Exp(2 * P.At(d, 0))
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			T.SubVec(gp.X.RowView(i), gp.X.RowView(j))
			T.DivElemVec(T, L)
			v := sigma_f * math.Exp(-0.5*T.Dot(T))
			gp.K.Set(i, j, v)
			if i == j {
				gp.Ky.Set(i, j, v+math.Exp(2*P.At(d+1, 0)))
			} else {
				gp.Ky.Set(i, j, v)
			}
		}

	}
	gp.P.CopyVec(P)
}

func (gp *Gp) logL(_P *mat64.Vector) float64 {
	gp.Cov(_P.RawVector().Data)
	Ky := gp.Ky
	alpha := &mat64.Vector{}
	alpha.SolveVec(Ky, gp.Y)
	n, _ := Ky.Dims()
	KySym := mat64.NewSymDense(n, Ky.RawMatrix().Data)
	chol := &mat64.Cholesky{}
	if ok := chol.Factorize(KySym); !ok {
		fmt.Println("logL: Ky matrix is not positive semi-definite.")
	}
	L := &mat64.TriDense{}
	L.LFromCholesky(chol)
	q := &mat64.Vector{}
	q.Diag(L)
	q.Log(q)
	return (q.Sum() + 0.5*alpha.Dot(gp.Y) + 0.5*float64(n)*math.Log(2*math.Pi))
}

func (gp *Gp) LogL() float64 {
	return gp.logL(gp.P)
}

func (gp *Gp) logLGrad(G, P *mat64.Vector) {
	if !P.Equals(gp.P) {
		gp.Cov(P.RawVector().Data)
	}
	K, Ky := gp.K, gp.Ky
	KyInv := &mat64.Dense{}
	err := KyInv.Inverse(gp.Ky)
	if err != nil {
		fmt.Println("logLGrad: matrix inversion error (%v)", err)
	}
	alpha := &mat64.Vector{}
	alpha.SolveVec(Ky, gp.Y)

	n, d := gp.X.Dims()

	W := &mat64.Dense{}
	Q := &mat64.Dense{}
	W.Outer(1, alpha, alpha)
	W.Sub(KyInv, W)
	Q.MulElem(W, K)
	D := mat64.NewDense(n, n, nil)
	DT := mat64.NewDense(n, n, nil)

	for j := 0; j < d; j++ {
		for i := 0; i < n; i++ {
			D.SetColVec(i, gp.X.ColView(j))
			DT.SetRowVec(i, gp.X.ColView(j))
		}
		DT.Sub(D, DT)
		DT.PowElem(DT, 2)
		D.MulElem(Q, DT)
		G.SetVec(j, mat64.Sum(D)/(2*math.Exp(gp.P.At(j, 0))))
	}

	G.SetVec(d, mat64.Sum(Q))
	G.SetVec(d+1, mat64.Trace(W)*math.Exp(2*gp.P.At(d+1, 0)))
}

func (gp *Gp) pCov(Xt *mat64.Dense) *mat64.Dense {
	n, d := gp.X.Dims()
	nt, _ := Xt.Dims()
	log_l := gp.P.ViewVec(0, d)
	l := mat64.NewVector(d, nil)
	q := mat64.NewVector(d, nil)
	l.Exp(log_l)
	sigma_f := math.Exp(2 * gp.P.At(d, 0))
	A := mat64.NewDense(n, nt, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < nt; j++ {
			q.SubVec(gp.X.RowView(i), Xt.RowView(j))
			q.DivElemVec(q, l)
			v := sigma_f * math.Exp(-0.5*q.Dot(q))
			A.Set(i, j, v)
		}
	}
	return A
}

func (gp *Gp) predict(Ks *mat64.Dense, mu, sigma *mat64.Vector) {
	_, nt := Ks.Dims()
	_, d := gp.X.Dims()
	sigma_f := math.Exp(2 * gp.P.At(d, 0))
	alpha := &mat64.Vector{}
	alpha.SolveVec(gp.Ky, gp.Y)
	mu.MulVec(Ks.T(), alpha)

	tmpMat := &mat64.Dense{}
	tmpMat.Solve(gp.Ky, Ks)
	tmpMat.MulElem(Ks, tmpMat)

	for i := 0; i < nt; i++ {
		sigma.SetVec(i, sigma_f-tmpMat.ColView(i).Sum())
	}
}

// calculate predictive means and noise free variance
func (gp *Gp) Predict(xt []float64) (_mu, _sigma []float64) {
	_, d := gp.X.Dims()
	nt := len(xt) / d
	Xt := mat64.NewDense(nt, d, xt)
	Ks := gp.pCov(Xt)
	_mu, _sigma = make([]float64, nt), make([]float64, nt)
	mu, sigma := mat64.NewVector(nt, _mu), mat64.NewVector(nt, _sigma)
	gp.predict(Ks, mu, sigma)
	return _mu, _sigma
}

func (o *GpOptimizer) Func(x []float64) float64 {
	return o.gp.logL(mat64.NewVector(len(x), x))
}

func (o *GpOptimizer) Grad(grad, x []float64) {
	o.gp.logLGrad(mat64.NewVector(len(grad), grad),
		mat64.NewVector(len(x), x))
}

func (aq *GpAquisition) Func(x []float64) float64 {
	_mu, _sigma := aq.gp.Predict(x)
	fx, sigma2 := _mu[0], _sigma[0]
	sigma := math.Sqrt(sigma2)
	if sigma < 0.0001 {
		sigma = 0.0001
	}
	u := (fx - aq.fx_max + 0.0001) / sigma
	z := (1 / math.Sqrt(2*math.Pi)) * math.Exp(-math.Pow(u, 2)/2)
	c := (0.5*math.Erf(u/math.Sqrt(2)) + 0.5)
	return -(sigma * (u*c + z))
}

func (aq *GpAquisition) Grad(grad, x []float64) {
	n, d := aq.gp.X.Dims()
	xt := mat64.NewDense(1, d, x)
	mu := mat64.NewVector(1, nil)
	sigma := mat64.NewVector(1, nil)
	Ks := aq.gp.pCov(xt)
	aq.gp.predict(Ks, mu, sigma)
	sigma.SetVec(0, math.Sqrt(sigma.At(0, 0)))
	// calculate Ks^T jacobian d Ks^T / dx
	log_l := aq.gp.P.ViewVec(0, d)
	Q := mat64.NewDense(d, n, nil)
	for i := 0; i < n; i++ {
		Q.ColView(i).SubVec(aq.gp.X.RowView(i), xt.RowView(0))
	}

	for i := 0; i < d; i++ {
		for j := 0; j < n; j++ {
			Q.Set(i, j, Q.At(i, j)*Ks.At(j, 0)*(1/math.Exp(2*log_l.At(i, 0))))
		}
	}

	// calculate d s(x) / dx
	alpha := mat64.NewVector(n, nil)
	tmp := mat64.NewVector(d, grad)
	ds_x := mat64.NewVector(d, nil)
	alpha.SolveVec(aq.gp.Ky, Ks.ColView(0))
	ds_x.MulVec(Q, alpha)
	ds_x.DivScalar(ds_x, sigma.At(0, 0))

	// calculate d u/ dx
	du_x := mat64.NewVector(d, nil)
	alpha.SolveVec(aq.gp.Ky, aq.gp.Y)
	du_x.MulVec(Q, alpha)
	u := (mu.At(0, 0) - aq.fx_max) / sigma.At(0, 0)
	z := (1 / math.Sqrt(2*math.Pi)) * math.Exp(-math.Pow(u, 2)/2)
	c := (0.5*math.Erf(u/math.Sqrt(2)) + 0.5)
	tmp.ScaleVec(u, ds_x)
	du_x.SubVec(du_x, tmp)
	du_x.DivScalar(du_x, sigma.At(0, 0))

	s1, s2 := (u*c + z), sigma.At(0, 0)*c
	ds_x.ScaleVec(s1, ds_x)
	du_x.ScaleVec(s2, du_x)
	tmp.AddVec(ds_x, du_x)
	tmp.ScaleVec(-1, tmp)
}

func (gp *Gp) optimize(x []float64, p optimize.Problem, method optimize.Method) []float64 {
	settings := optimize.DefaultSettings()
	settings.Recorder = nil

	if method != nil && method.Needs().Gradient {
		// Turn off function convergence checks for gradient-based methods.
		settings.FunctionConverge = nil
	} else {
		settings.FunctionConverge.Iterations = 50
		settings.FunctionConverge.Absolute = 1e-12
	}
	settings.GradientThreshold = 1e-20

	result, err := optimize.Local(p, x, settings, method)
	if err != nil {
		fmt.Println("error finding minimum (%v)", err)
	}
	if result == nil {
		fmt.Println("nil result without error")
	}
	return result.Location.X
}

func (gp *Gp) OptParams(x []float64) []float64 {
	p := optimize.Problem{
		Func: gp.opt.Func,
		Grad: gp.opt.Grad,
	}
	Po := gp.optimize(x, p, &optimize.BFGS{})
	gp.Cov(Po)
	return Po
}

func (gp *Gp) InitAcquisition() {
	// locate fmax
	fmax := math.SmallestNonzeroFloat64
	idx := 0
	for i := 0; i < gp.Y.Len(); i++ {
		if gp.Y.At(i, 0) > fmax {
			fmax = gp.Y.At(i, 0)
			idx = i
		}
	}
	gp.acq.x_max = gp.X.RowView(idx).RawVector().Data
	_mu, _ := gp.Predict(gp.acq.x_max)
	gp.acq.fx_max = _mu[0]
}
func (gp *Gp) OptExpImp(x []float64) []float64 {
	p := optimize.Problem{
		Func: gp.acq.Func,
		Grad: gp.acq.Grad,
	}
	return gp.optimize(x, p, &optimize.NelderMead{})
}

func (gp *DynGp) Add(x []float64, fx float64, dd interface{}) {
	// just add new row
	gp.bdata.X.SetRow(gp.bdata.nc, x)
	gp.bdata.Y.SetVec(gp.bdata.nc, fx)
	gp.bdata.D[gp.bdata.nc] = dd
	gp.GrowOneRow()
}

func (gp *DynGp) InitAcquisition() {
	gp.acq.x_max = gp.X.RowView(gp.best).RawVector().Data
	_mu, _ := gp.Predict(gp.acq.x_max)
	gp.acq.fx_max = _mu[0]
}

func (gp *DynGp) GrowOneRow() {
	// just add new row
	gp.bdata.nc = gp.bdata.nc + 1
	gp.X = gp.bdata.X.View(0, 0, gp.bdata.nc, gp.bdata.dc).(*mat64.Dense)
	gp.Y = gp.bdata.Y.ViewVec(0, gp.bdata.nc)
	gp.K.Resize(gp.bdata.nc, gp.bdata.nc)
	gp.Ky.Resize(gp.bdata.nc, gp.bdata.nc)
}

func (gp *DynGp) UpdateBest(i int) {
	if gp.best != -1 && gp.best == i {
		fmax := math.SmallestNonzeroFloat64
		for i := 0; i < gp.Y.Len(); i++ {
			if gp.Y.At(i, 0) > fmax {
				fmax = gp.Y.At(i, 0)
				gp.best = i
			}
		}
		return
	}
	if gp.best == -1 || gp.bdata.Y.At(i, 0) > gp.bdata.Y.At(gp.best, 0) {
		gp.best = i
	}
}

func (gp *DynGp) Delete(i int) {
	// delete the given row
	if gp.mark1 == gp.bdata.nc {
		gp.mark1 = gp.bdata.nc - 1
	}
	if gp.mark1 == i {
		gp.mark1 = -1
	}
	if gp.mark1 == gp.bdata.nc-1 {
		gp.mark1 = i
	}
	if gp.mark2 == gp.bdata.nc {
		gp.mark2 = gp.bdata.nc - 1
	}
	if gp.mark2 == i {
		gp.mark2 = -1
	}
	if gp.mark2 == gp.bdata.nc-1 {
		gp.mark2 = i
	}
	if gp.best == gp.bdata.nc-1 {
		gp.best = i
	}
	gp.bdata.nc = gp.bdata.nc - 1
	gp.bdata.X.SetRowVec(i, gp.bdata.X.RowView(gp.bdata.nc))
	gp.bdata.Y.SetVec(i, gp.bdata.Y.At(gp.bdata.nc, 0))
	gp.bdata.D[i] = gp.bdata.D[gp.bdata.nc]
	gp.X = gp.bdata.X.View(0, 0, gp.bdata.nc, gp.bdata.dc).(*mat64.Dense)
	gp.Y = gp.bdata.Y.ViewVec(0, gp.bdata.nc)
	gp.K.Resize(gp.bdata.nc, gp.bdata.nc)
	gp.Ky.Resize(gp.bdata.nc, gp.bdata.nc)
}

func (gp *DynGp) DeleteMinY() {
	// find MinY
	fmin := math.MaxFloat64
	idx := 0
	for i := 0; i < gp.Y.Len(); i++ {
		if gp.Y.At(i, 0) < fmin {
			fmin = gp.Y.At(i, 0)
			idx = i
		}
	}
	gp.Delete(idx)
}

func (gp *DynGp) Size() int {
	return gp.bdata.nc
}
