package baysopt

import (
	"fmt"
	"gem"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

type Qstatistics struct {
	qa float64 // quality for active settings
	na int32   // number of evaluations for active setting
}
type BlockModel struct {
	gp              *DynGp // dynmic gausian process (supports addition and deletion of data points)
	interval        int    // timer to control exploration of parameter space
	interval_ctr    int
	n_max           int
	Init, Low, High []float64
	act_stats       *Qstatistics
	act_sett        *mat64.Vector
	action          int
}

func NewBlockModel(n_max, d int, interval int, init, low, high []float64) *BlockModel {
	gp := NewDynGp(n_max, d)
	gp.mark1 = 0
	gp.best = 0
	for i := 0; i < n_max; i++ {
		gp.bdata.D[i] = &Qstatistics{0, 0}
	}
	return &BlockModel{gp: gp, interval: interval, interval_ctr: 0, n_max: n_max, Init: init, Low: low, High: high,
		act_sett: mat64.NewVector(d, init), act_stats: &Qstatistics{0, 0}, action: 1}
}

func (bm *BlockModel) UpdateActiveSetting(x []float64, q float64) {
	// check if x is the active setting
	i_sett_vec := mat64.NewVector(len(x), x)
	if i_sett_vec.Equals(bm.act_sett) {
		bm.act_stats.qa += q
		bm.act_stats.na += 1
		bm.interval_ctr = bm.interval_ctr + 1
	}
	if bm.gp.mark2 != -1 {
		if i_sett_vec.Equals(bm.gp.bdata.X.RowView(bm.gp.mark2)) {
			s := bm.gp.bdata.D[bm.gp.mark2].(*Qstatistics)
			s.qa += q
			s.na += 1
			bm.gp.bdata.Y.SetVec(bm.gp.mark2, s.qa/float64(s.na))
		}
	}
	fmt.Printf("%v\n\n", bm.gp.Y)
}

func (bm *BlockModel) GetActiveSetting() []float64 {

	if bm.interval_ctr%bm.interval == bm.interval-1 {
		// update at mark
		s := bm.gp.bdata.D[bm.gp.mark1].(*Qstatistics)
		s.na += bm.act_stats.na
		s.qa += bm.act_stats.qa
		bm.act_stats.na, bm.act_stats.qa = 0, 0
		bm.gp.bdata.X.SetRow(bm.gp.mark1, bm.act_sett.RawVector().Data)
		bm.gp.bdata.Y.SetVec(bm.gp.mark1, s.qa/float64(s.na))
		bm.gp.UpdateBest(bm.gp.mark1)
		fmt.Printf("best=%v  best_value=%v \n\n", bm.gp.best, bm.gp.bdata.Y.At(bm.gp.best, 0))

		if bm.gp.Size() < bm.n_max-2 {
			bm.gp.GrowOneRow()
		}
		if bm.gp.Size() < bm.n_max-2 {
			// Initialization by random guesses
			bm.gp.mark2 = bm.gp.mark1
			bm.gp.mark1 = bm.gp.Size()
			bm.RandomSample(bm.act_sett.RawVector().Data)
		} else if bm.action == 1 {
			// Baysian optimization to select next best guess
			P := gem.Point{1, 1, 0.1, 0.2}
			P.Apply(func(i int, v float64) float64 { return math.Log(v) })
			bm.gp.OptParams(P)
			bm.gp.InitAcquisition()
			sel := int(rand.Float32() * float32(bm.gp.Size()))
			xn := bm.gp.OptExpImp(bm.gp.bdata.X.RowView(sel).RawVector().Data)
			fmt.Printf("xn := %v", xn)
			copy(bm.act_sett.RawVector().Data, xn)
			bm.gp.GrowOneRow()
			bm.gp.mark2 = bm.gp.mark1
			bm.gp.mark1 = bm.gp.Size() - 1
			bm.action = 2
		} else if bm.action == 2 {
			// Random re-evaluation of a previous guess
			bm.gp.mark2 = bm.gp.mark1
			bm.gp.mark1 = int(rand.Float32() * float32(bm.gp.Size()-1))
			bm.act_sett.CopyVec(bm.gp.bdata.X.RowView(bm.gp.mark1))
			bm.action = 3
		} else {
			// Random exploration by a new guess
			bm.gp.DeleteMinY()
			bm.gp.DeleteMinY()
			bm.gp.mark2 = bm.gp.mark1
			bm.gp.mark1 = bm.gp.Size()
			bm.RandomSample(bm.act_sett.RawVector().Data)
			s := bm.gp.bdata.D[bm.gp.mark1].(*Qstatistics)
			s.na, s.qa = 0, 0
			bm.action = 1
		}
		bm.interval_ctr = 0
	}

	return bm.act_sett.RawVector().Data
}

func (bm *BlockModel) RandomSample(x []float64) {
	for i := 0; i < len(x); i++ {
		x[i] = rand.Float64()*(bm.High[i]-bm.Low[i]) + bm.Low[i]
	}
}

func (bm *BlockModel) GetBestSetting() ([]float64, bool) {
	if bm.gp.best >= 0 {
		return bm.gp.bdata.X.RowView(bm.gp.best).RawVector().Data, true
	}
	return nil, false
}

func (bm *BlockModel) Dispose() {}
