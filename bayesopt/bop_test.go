package baysopt

import (
	// "fmt"
	"gem"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"testing"
)

func TestNewGp(t *testing.T) {
	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	gp := NewGp(x, fx)
	gp.X.Apply(func(r, c int, v float64) float64 { return x[r*1+c] - gp.X.At(r, c) }, gp.X)
	gp.Y.Apply(func(i int, v float64) float64 { return fx[i] - gp.Y.At(i, 0) }, gp.Y)
	n, d := gp.X.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			if gp.X.At(i, j) != 0 {
				t.Errorf("Error setting the data array")
			}
		}
	}
	for i := 0; i < n; i++ {
		if gp.Y.At(i, 0) != 0 {
			t.Errorf("Error setting the target vector")
		}
	}
}

func TestCov(t *testing.T) {
	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}

	C := gem.Point{
		0.0100000000000000, 9.64529963139591e-26, 1.99167935913507e-16, 6.99621772456007e-08, 0.00129399773798042, 7.76224806971967e-11, 7.23779185139208e-06,
		9.64529963139591e-26, 0.0100000000000000, 0.000629161746301273, 1.10650872942684e-52, 1.13304074540403e-35, 1.58064148368764e-06, 6.92773061326821e-46,
		1.99167935913507e-16, 0.000629161746301273, 0.0100000000000000, 2.16682219461150e-38, 2.72175507631303e-24, 0.00186760392920930, 1.09649659640960e-32,
		6.99621772456007e-08, 1.10650872942684e-52, 2.16682219461150e-38, 0.0100000000000000, 0.000172225620957827, 6.35323839161656e-29, 0.00564443423221903,
		0.00129399773798042, 1.13304074540403e-35, 2.72175507631303e-24, 0.000172225620957827, 0.0100000000000000, 4.31049141453935e-17, 0.00204892837586956,
		7.76224806971967e-11, 1.58064148368764e-06, 0.00186760392920930, 6.35323839161656e-29, 4.31049141453935e-17, 0.0100000000000000, 4.53212636617284e-24,
		7.23779185139208e-06, 6.92773061326821e-46, 1.09649659640960e-32, 0.00564443423221903, 0.00204892837586956, 4.53212636617284e-24, 0.0100000000000000,
	}
	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	n, _ := gp.K.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(gp.K.At(i, j)-C[i*n+j]) > 1e-12 {
				t.Errorf("Error setting the covariance array at (%v,%v) = %v, found %v", i, j, C[i*n+j], gp.K.At(i, j))
			}
		}
	}
}

func TestLogL(t *testing.T) {
	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}

	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	l := gp.LogL()
	if math.Abs(l+3.8677) > 1e-4 {
		t.Errorf("Error calculating Log likelihood: expected %v, found %v", -3.8677, l)
	}
}

func TestGrad(t *testing.T) {

	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	G := gem.Point{
		-0.0137286352284538,
		1.30098260615498,
		5.31277097124521,
	}

	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	Gr := mat64.NewVector(gp.P.Len(), nil)
	gp.logLGrad(Gr, gp.P)
	for i := 0; i < Gr.Len(); i++ {
		if (math.Abs(Gr.At(i, 0) - G[i])) > 1e-10 {
			t.Errorf("Error calculating likelihood gradient: expected %v, found %v", G[i], Gr.At(i, 0))
		}
	}

}

func TestOptimizeParams(t *testing.T) {

	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	Pt := gem.Point{
		-0.592696129373652,
		-2.95277980316074,
		-13.8343749960392,
	}
	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	Pm := gp.OptParams(P)
	for i := 0; i < len(P); i++ {
		if (math.Abs(Pt[i] - Pm[i])) > 1e-5 {
			if Pt[i] < Pm[i] {
				t.Errorf("Error calculating optimum Parameters: expected %v, found %v", Pt[i], Pm[i])
			}
		}
	}
	// test if gp is updated to final parameters
	for i := 0; i < len(Pm); i++ {
		if math.Abs(Pm[i]-gp.P.At(i, 0)) != 0 {
			t.Errorf("Error setting the gaussian process params to the final optimim ones: expected %v, found %v", Pm[i], gp.P.At(i, 0))
		}
	}

}

func TestPredict(t *testing.T) {

	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	xt := gem.Point{10.9, -5}
	Ft := gem.Point{
		-5.91546891713912e-29,
		-0.00122801475624516,
	}
	St := gem.Point{
		0.00272425684274105,
		0.00270718350807919,
	}
	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	gp.OptParams(P)
	Fm, Sm := gp.Predict(xt)
	for i := 0; i < len(xt); i++ {
		if (math.Abs(Ft[i] - Fm[i])) > 1e-5 {
			t.Errorf("Error calculating predicted mean: expected %v, found %v", Ft[i], Fm[i])
		}
		if (math.Abs(St[i] - Sm[i])) > 1e-5 {
			t.Errorf("Error calculating predicted variance: expected %v, found %v", Ft[i], Fm[i])
		}
	}

}

func TestEiAquisition(t *testing.T) {

	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	xt := gem.Point{10.9, -5}
	Ei := gem.Point{
		-0.00172608186452199,
		-0.00161508372944050,
	}
	GtEi := gem.Point{
		-8.82685353713834e-29,
		-0.000171157358437755,
	}

	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	gp.OptParams(P)
	gp.InitAcquisition()
	xx, gx := []float64{0}, []float64{0}
	for i := 0; i < len(xt); i++ {
		xx[0] = xt[i]
		fei := gp.acq.Func(xx)
		if (math.Abs(Ei[i] - fei)) > 1e-5 {
			t.Errorf("Error calculating expected improvement: expected %v, found %v", Ei[i], fei)
		}
		gp.acq.Grad(gx, xx)
		if (math.Abs(gx[0] - GtEi[i])) > 1e-5 {
			t.Errorf("Error calculating gradient of expected improvement: expected %v, found %v", GtEi[i], gx[0])
		}
	}
}

func TestOptExpImp(t *testing.T) {

	x := gem.Point{
		-0.1338,
		-10.4290,
		-8.0770,
		4.7386,
		1.8885,
		-6.2451,
		3.6691}
	fx := gem.Point{
		0.0058,
		-0.0953,
		0.0756,
		-0.0546,
		0.0313,
		-0.0152,
		-0.0023}
	xi := gem.Point{-8.0770}
	xo := gem.Point{-8.45580656738281}

	gp := NewGp(x, fx)
	P := gem.Point{1, 0.1, 0.2}
	P.Apply(func(i int, v float64) float64 { return math.Log(v) })
	gp.Cov(P)
	gp.OptParams(P)
	gp.InitAcquisition()
	xn := gp.OptExpImp(xi)
	fn := Fxx(xn[0])
	fo := Fxx(xo[0])
	if (math.Abs(fn - fo)) > 0.1 {
		t.Errorf("Error calculating next optimum location using EI: expected %v, found %v", xo[0], xn[0])
	}
}
func Fxx(x float64) float64 {
	r := rand.New(rand.NewSource(99))

	return (x*math.Sin(x) + r.NormFloat64()) / 100

}
