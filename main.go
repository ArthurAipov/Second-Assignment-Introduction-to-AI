// Package main implements a simple evolutionary algorithm that tries
// to approximate input images using overlapping letters A, R, T, H, U
// as "building blocks". It also generates plots of fitness over generations.
package main

import (
	"bufio"
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// RunData stores numeric data parsed from one log file.
// It is later used to draw a plot of fitness vs generations.
type RunData struct {
	Gen  []float64 // Generation index
	Best []float64 // Best (here: global) fitness per generation
	Avg  []float64 // Average fitness per generation
}

// Global configuration constants for the evolutionary algorithm.
const (
	// PopSize is the number of individuals (letters) in the population.
	PopSize = 9000

	// Generations is the maximum number of generations for each run.
	Generations = 500

	// MutationStart is the mutation rate at the beginning.
	MutationStart = 0.6

	// MutationEnd is the mutation rate at the end.
	MutationEnd = 0.15

	// PosterizeLevels is the number of color levels used in posterization.
	PosterizeLevels = 7

	// EdgeThreshold is the gradient threshold for edge drawing.
	EdgeThreshold = 120.0

	// MaxDuration is a time limit for one evolutionary run.
	MaxDuration = 2 * time.Minute

	// NumRuns is how many independent runs are done per image.
	NumRuns = 3

	// TopK is how many best populations (by global fitness) we remember.
	TopK = 5
)

// inputs lists all input image paths that will be processed.
var inputs = []string{
	"inputs/input1.jpg",
	"inputs/input2.jpg",
	"inputs/input3.jpg",
	"inputs/input4.jpg",
	"inputs/input5.jpg",
	"inputs/input6.jpg",
}

// Below we define 9x9 boolean masks for each letter.
// true = pixel is part of the letter, false = background.

var letterA = [][]bool{
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, true, true, true, false, false, false},
	{false, false, true, true, false, true, true, false, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{true, true, true, true, true, true, true, true, true},
	{true, false, false, false, false, false, false, false, true},
	{true, false, false, false, false, false, false, false, true},
	{true, false, false, false, false, false, false, false, true},
}

var letterR = [][]bool{
	{false, false, true, true, true, true, true, false, false},
	{false, false, true, false, false, false, true, false, false},
	{false, false, true, false, false, false, true, false, false},
	{false, false, true, false, false, false, true, false, false},
	{false, false, true, true, true, true, false, false, false},
	{false, false, true, false, true, false, false, false, false},
	{false, false, true, false, false, true, false, false, false},
}
var letterT = [][]bool{
	{false, true, true, true, true, true, true, true, false},
	{false, true, true, true, true, true, true, true, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
	{false, false, false, false, true, false, false, false, false},
}

var letterH = [][]bool{
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, true, true, true, true, true, true, true},
	{false, true, true, true, true, true, true, true, true},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
}

var letterU = [][]bool{
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, false, false, false, false, false, true, false},
	{false, true, true, false, false, false, true, true, false},
	{false, false, true, true, true, true, true, false, false},
}

// init seeds the global random number generator.
// This makes the algorithm behave differently on each run.
func init() {
	rand.Seed(time.Now().UnixNano())
}

// LoadJPEG opens the file at the given path and decodes it as a JPEG image.
// It returns the decoded image or an error if something goes wrong.
func LoadJPEG(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		return nil, err
	}
	return img, nil
}

// SavePNG writes the given image as a JPEG file to the given path.
// Despite the function name, the file is saved using JPEG encoding.
// The quality is set to 95.
func SavePNG(img image.Image, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return jpeg.Encode(f, img, &jpeg.Options{Quality: 95})
}

// Blur applies a simple box blur with the given radius.
// Radius is the half-size of the kernel. If radius <= 0, the image is returned unchanged.
func Blur(src image.Image, radius int) image.Image {
	if radius <= 0 {
		return src
	}
	b := src.Bounds()
	dst := image.NewRGBA(b)

	// Copy border pixels without blur, to avoid out-of-bounds access.
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if x < b.Min.X+radius || x >= b.Max.X-radius ||
				y < b.Min.Y+radius || y >= b.Max.Y-radius {
				dst.Set(x, y, src.At(x, y))
			}
		}
	}

	// Number of pixels in the blur kernel.
	size := (2*radius + 1) * (2*radius + 1)

	// Blur inner region.
	for y := b.Min.Y + radius; y < b.Max.Y-radius; y++ {
		for x := b.Min.X + radius; x < b.Max.X-radius; x++ {
			var sr, sg, sb uint32
			for dy := -radius; dy <= radius; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					r, g, bl, _ := src.At(x+dx, y+dy).RGBA()
					sr += r
					sg += g
					sb += bl
				}
			}
			r := uint8((sr / uint32(size)) >> 8)
			g := uint8((sg / uint32(size)) >> 8)
			bb := uint8((sb / uint32(size)) >> 8)
			dst.Set(x, y, color.RGBA{r, g, bb, 255})
		}
	}

	return dst
}

// Posterize reduces the number of distinct color levels in the image.
// For each channel, the color is mapped to one of 'levels' possible values.
func Posterize(src image.Image, levels int) image.Image {
	if levels < 2 {
		levels = 2
	}
	b := src.Bounds()
	dst := image.NewRGBA(b)

	// Distance between quantization levels in [0..255].
	step := 255.0 / float64(levels-1)

	// quantize maps one byte (0..255) to the nearest quantization level.
	quantize := func(v uint8) uint8 {
		idx := float64(v) / 255.0 * float64(levels-1)
		roundIdx := int(idx + 0.5)
		if roundIdx < 0 {
			roundIdx = 0
		}
		if roundIdx > levels-1 {
			roundIdx = levels - 1
		}
		return uint8(step * float64(roundIdx))
	}

	// Apply quantization to each pixel.
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r16, g16, b16, a16 := src.At(x, y).RGBA()
			r := uint8(r16 >> 8)
			g := uint8(g16 >> 8)
			bb := uint8(b16 >> 8)
			a := uint8(a16 >> 8)

			rq := quantize(r)
			gq := quantize(g)
			bq := quantize(bb)

			dst.Set(x, y, color.RGBA{rq, gq, bq, a})
		}
	}
	return dst
}

// DrawEdges runs a Sobel-like edge detector and darkens pixels that
// have gradient magnitude higher than 'threshold'.
// If threshold <= 0, the original image is returned.
func DrawEdges(src image.Image, threshold float64) image.Image {
	if threshold <= 0 {
		return src
	}

	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(b)

	// Convert to grayscale first, for edge detection.
	gray := make([][]float64, h)
	for y := 0; y < h; y++ {
		gray[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			r16, g16, b16, _ := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			r := float64(r16 >> 8)
			g := float64(g16 >> 8)
			bb := float64(b16 >> 8)
			gray[y][x] = 0.299*r + 0.587*g + 0.114*bb
		}
	}

	// Sobel kernels.
	gx := [][]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	gy := [][]int{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1},
	}

	// For each pixel, compute gradient and darken if above threshold.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r16, g16, b16, a16 := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			r := uint8(r16 >> 8)
			g := uint8(g16 >> 8)
			bb := uint8(b16 >> 8)
			a := uint8(a16 >> 8)

			// Skip borders to avoid out-of-bounds.
			if x > 0 && x < w-1 && y > 0 && y < h-1 {
				var grayX, grayY float64
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						val := gray[y+dy][x+dx]
						grayX += val * float64(gx[dy+1][dx+1])
						grayY += val * float64(gy[dy+1][dx+1])
					}
				}
				grad := math.Sqrt(grayX*grayX + grayY*grayY)
				if grad > threshold {
					// Darken the pixel if it lies on an edge.
					r, g, bb = 10, 10, 10
				}
			}
			dst.Set(b.Min.X+x, b.Min.Y+y, color.RGBA{r, g, bb, a})
		}
	}
	return dst
}

// GeneLetter represents parameters of one drawn letter (one "gene").
// Position, scale and color are stored here.
type GeneLetter struct {
	X, Y       int   // Top-left coordinate where the letter is drawn
	Scale      int   // Size multiplier for the letter mask
	R, G, B, A uint8 // Color and alpha of the letter
}

// Individual represents one letter in the population, with a character
// (A/R/T/H/U), its drawing parameters, and a fitness value.
type Individual struct {
	Ch      rune       // Which letter (A/R/T/H/U)
	Gens    GeneLetter // Drawing parameters for the letter
	Fitness float64    // Fitness value of this individual
}

// LetterMask returns the 9x9 boolean mask for the given character.
// If the character is unknown, it falls back to the mask of 'A'.
func LetterMask(ch rune) [][]bool {
	switch ch {
	case 'A':
		return letterA
	case 'R':
		return letterR
	case 'T':
		return letterT
	case 'H':
		return letterH
	case 'U':
		return letterU
	default:
		return letterA
	}
}

// RandomLetter creates a new Individual with random letter type,
// random position, random color and random scale.
// The values are chosen so that the letter remains inside the image bounds.
func RandomLetter(width, height int) *Individual {
	chars := []rune{'A', 'R', 'T', 'H', 'U'}
	ch := chars[rand.Intn(len(chars))]

	mask := LetterMask(ch)
	maskH := len(mask)
	maskW := len(mask[0])

	// Random scale in a small range.
	scale := 2 + rand.Intn(3) // 2..4 inclusive

	// Compute max X/Y so that the letter mask with scale fits on the image.
	maxX := width - maskW*scale
	maxY := height - maskH*scale
	if maxX < 0 {
		maxX = 0
	}
	if maxY < 0 {
		maxY = 0
	}

	// Random top-left position.
	x := 0
	y := 0
	if maxX > 0 {
		x = rand.Intn(maxX)
	}
	if maxY > 0 {
		y = rand.Intn(maxY)
	}

	// Random color with alpha in a mid-to-high range.
	r := uint8(rand.Intn(256))
	g := uint8(rand.Intn(256))
	bb := uint8(rand.Intn(256))
	alpha := uint8(160 + rand.Intn(96)) // 160..255

	return &Individual{
		Ch: ch,
		Gens: GeneLetter{
			X:     x,
			Y:     y,
			Scale: scale,
			R:     r,
			G:     g,
			B:     bb,
			A:     alpha,
		},
		Fitness: 0,
	}
}

// Mix does simple alpha blending of src over dst.
// The returned color is the blended result.
func Mix(dst, src color.RGBA) color.RGBA {
	af := float64(src.A) / 255.0
	if af <= 0 {
		return dst
	}
	inv := 1.0 - af
	r := uint8(af*float64(src.R) + inv*float64(dst.R))
	g := uint8(af*float64(src.G) + inv*float64(dst.G))
	b := uint8(af*float64(src.B) + inv*float64(dst.B))
	return color.RGBA{r, g, b, 255}
}

// DrawLetter draws the given Individual (letter) on the given RGBA image.
// The letter is scaled and positioned according to its GeneLetter data.
func DrawLetter(grid *image.RGBA, ind *Individual) {
	mask := LetterMask(ind.Ch)
	if mask == nil {
		return
	}

	b := grid.Bounds()
	width, height := b.Dx(), b.Dy()

	maskH := len(mask)
	if maskH == 0 {
		return
	}
	maskW := len(mask[0])

	for my := 0; my < maskH; my++ {
		for mx := 0; mx < maskW; mx++ {
			if !mask[my][mx] {
				continue
			}

			// Starting position of this mask cell, scaled.
			startX := ind.Gens.X + mx*ind.Gens.Scale
			startY := ind.Gens.Y + my*ind.Gens.Scale

			// Fill a small rectangle for each mask cell, based on scale.
			for dy := 0; dy < ind.Gens.Scale; dy++ {
				for dx := 0; dx < ind.Gens.Scale; dx++ {
					px := startX + dx
					py := startY + dy

					if px < 0 || px >= width || py < 0 || py >= height {
						continue
					}

					old := grid.RGBAAt(px, py)
					src := color.RGBA{
						R: ind.Gens.R,
						G: ind.Gens.G,
						B: ind.Gens.B,
						A: ind.Gens.A,
					}
					newCol := Mix(old, src)
					grid.SetRGBA(px, py, newCol)
				}
			}
		}
	}
}

// RenderPopulation renders the whole population into a new RGBA image.
// The alpha of each individual is re-scaled based on its fitness,
// so better individuals are more visible.
func RenderPopulation(pop []*Individual, width, height int) *image.RGBA {
	grid := image.NewRGBA(image.Rect(0, 0, width, height))
	if len(pop) == 0 {
		return grid
	}

	// Make a copy of the population and sort by fitness.
	sorted := make([]*Individual, len(pop))
	copy(sorted, pop)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Fitness < sorted[j].Fitness
	})

	minF := sorted[0].Fitness
	maxF := sorted[len(sorted)-1].Fitness
	if maxF == minF {
		// If all fitness values are the same, draw them as they are.
		for _, ind := range sorted {
			DrawLetter(grid, ind)
		}
		return grid
	}

	// Map fitness to alpha range. Best fitness gets highest alpha.
	for _, ind := range sorted {
		q := (ind.Fitness - minF) / (maxF - minF)
		alpha := uint8(40 + q*200)

		tmp := *ind
		tmp.Gens.A = alpha
		DrawLetter(grid, &tmp)
	}

	return grid
}

// RenderPopulationRaw renders the population into a new RGBA image
// using the individuals' own alpha values, without re-scaling by fitness.
func RenderPopulationRaw(pop []*Individual, width, height int) *image.RGBA {
	grid := image.NewRGBA(image.Rect(0, 0, width, height))
	for _, ind := range pop {
		DrawLetter(grid, ind)
	}
	return grid
}

// LetterFitness computes the fitness of one Individual with respect to
// the target image. The fitness is the negative mean squared error (MSE)
// between the individual's color and the target pixels that it covers.
// More positive values are better.
func LetterFitness(ind *Individual, target image.Image) float64 {
	mask := LetterMask(ind.Ch)
	if mask == nil {
		return -1e18
	}
	b := target.Bounds()
	width, height := b.Dx(), b.Dy()

	maskH := len(mask)
	if maskH == 0 {
		return -1e18
	}
	maskW := len(mask[0])

	var err float64
	var count int

	for my := 0; my < maskH; my++ {
		for mx := 0; mx < maskW; mx++ {
			if !mask[my][mx] {
				continue
			}

			startX := ind.Gens.X + mx*ind.Gens.Scale
			startY := ind.Gens.Y + my*ind.Gens.Scale

			for dy := 0; dy < ind.Gens.Scale; dy++ {
				for dx := 0; dx < ind.Gens.Scale; dx++ {
					px := startX + dx
					py := startY + dy

					if px < 0 || px >= width || py < 0 || py >= height {
						continue
					}

					// Target pixel color.
					tr16, tg16, tb16, _ := target.At(b.Min.X+px, b.Min.Y+py).RGBA()
					tr := float64(tr16 >> 8)
					tg := float64(tg16 >> 8)
					tb := float64(tb16 >> 8)

					// Current individual's color.
					rr := float64(ind.Gens.R)
					rg := float64(ind.Gens.G)
					rb := float64(ind.Gens.B)

					// Squared difference in RGB space.
					dr := rr - tr
					dg := rg - tg
					db := rb - tb

					err += dr*dr + dg*dg + db*db
					count++
				}
			}
		}
	}

	if count == 0 {
		// If letter is completely out of bounds, strongly punish.
		return -1e18
	}
	// We return negative average error (so bigger is better).
	return -err / float64(count)
}

// GlobalFitness returns the sum of fitness of all individuals in the population.
func GlobalFitness(pop []*Individual) float64 {
	var s float64
	for _, ind := range pop {
		s += ind.Fitness
	}
	return s
}

// Mutate changes the given Individual with the given mutation rate.
// If colorOnly is true, position and scale are not changed, only color.
// Otherwise, all gene fields can change (within some bounds).
func Mutate(ind *Individual, width, height int, rate float64, colorOnly bool) {
	// With probability (1-rate) do nothing.
	if rand.Float64() >= rate {
		return
	}

	// clampU8 keeps a value within [0..255].
	clampU8 := func(v int) uint8 {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return uint8(v)
	}

	g := &ind.Gens

	if !colorOnly {
		// Small random move of the letter.
		g.X += rand.Intn(7) - 3
		g.Y += rand.Intn(7) - 3

		// Clamp position so it stays inside image.
		if g.X < 0 {
			g.X = 0
		}
		if g.Y < 0 {
			g.Y = 0
		}
		if g.X >= width {
			g.X = width - 1
		}
		if g.Y >= height {
			g.Y = height - 1
		}

		// Change scale in a relatively large range, then clamp.
		g.Scale += rand.Intn(100) - 1
		if g.Scale < 2 {
			g.Scale = 2
		}
		if g.Scale > 8 {
			g.Scale = 8
		}
	}

	// Small random change of RGB channels.
	g.R = clampU8(int(g.R) + rand.Intn(21) - 10)
	g.G = clampU8(int(g.G) + rand.Intn(21) - 10)
	g.B = clampU8(int(g.B) + rand.Intn(21) - 10)

	// Small random change of alpha channel.
	g.A = clampU8(int(g.A) + rand.Intn(31) - 15)
	if g.A < 80 {
		g.A = 80
	}
	if g.A > 220 {
		g.A = 220
	}

	// With small probability, scale all channels together (like brightness).
	if rand.Float64() < 0.1 {
		f := 0.8 + rand.Float64()*0.4 // factor in [0.8..1.2]
		g.R = clampU8(int(float64(g.R) * f))
		g.G = clampU8(int(float64(g.G) * f))
		g.B = clampU8(int(float64(g.B) * f))
	}

	// With very small probability, change the letter character itself.
	if rand.Float64() < 0.01 {
		chars := []rune{'A', 'R', 'T', 'H', 'U'}
		ind.Ch = chars[rand.Intn(len(chars))]
	}
}

// CloneIndividual creates a deep copy of the given Individual.
// The new individual has the same values but is a different object in memory.
func CloneIndividual(ind *Individual) *Individual {
	return &Individual{
		Ch: ind.Ch,
		Gens: GeneLetter{
			X:     ind.Gens.X,
			Y:     ind.Gens.Y,
			Scale: ind.Gens.Scale,
			R:     ind.Gens.R,
			G:     ind.Gens.G,
			B:     ind.Gens.B,
			A:     ind.Gens.A,
		},
		Fitness: ind.Fitness,
	}
}

// InitPopulation creates a new population of size popSize.
// Each individual is random and its fitness is computed immediately.
func InitPopulation(popSize, width, height int, target image.Image) []*Individual {
	pop := make([]*Individual, popSize)
	for i := 0; i < popSize; i++ {
		ind := RandomLetter(width, height)
		ind.Fitness = LetterFitness(ind, target)
		pop[i] = ind
	}
	return pop
}

// ClonePopulation creates a deep copy of the whole population.
func ClonePopulation(pop []*Individual) []*Individual {
	clone := make([]*Individual, len(pop))
	for i, ind := range pop {
		clone[i] = CloneIndividual(ind)
	}
	return clone
}

// Evolve runs the evolutionary algorithm for the given number of generations
// on the given target image. It logs progress to the given logSource file,
// saves intermediate images, and finally returns the last population.
func Evolve(target image.Image, generations int, logSource string) []*Individual {
	logFile, err := os.Create(logSource)
	if err != nil {
		panic(err)
	}
	defer logFile.Close()

	b := target.Bounds()
	width, height := b.Dx(), b.Dy()

	// Initialize population and compute global fitness.
	pop := InitPopulation(PopSize, width, height, target)
	globalFit := GlobalFitness(pop)

	// Save initial raw population rendering.
	prefix_init := strings.TrimSuffix(logSource, ".txt")
	rawInitPath := fmt.Sprintf("%s_gen_init_raw.jpg", prefix_init)
	rawImg := RenderPopulationRaw(pop, width, height)
	_ = SavePNG(rawImg, rawInitPath)
	fmt.Println("saved raw init canvas:", rawInitPath)

	start := time.Now()

	// Track TopK best global fitness values and populations.
	bestFits := make([]float64, TopK)
	bestGens := make([]int, TopK)
	bestPops := make([][]*Individual, TopK)
	for i := range bestFits {
		bestFits[i] = math.Inf(-1)
	}

	for gen := 0; gen < generations; gen++ {
		// Stop early if time limit is reached.
		if time.Since(start) > MaxDuration {
			fmt.Fprintf(logFile, "Time limit reached, stopping at generation %d\n", gen)
			fmt.Println("Time limit reached, stopping at generation", gen)
			break
		}

		// For early generations we only mutate color, not position.
		colorOnly := gen < generations/4

		// Mutation rate changes linearly from MutationStart to MutationEnd.
		progress := float64(gen) / float64(generations-1)
		curMutation := MutationStart + (MutationEnd-MutationStart)*progress

		// Loop over population and try to mutate each individual.
		for i := range pop {
			old := pop[i]
			oldFit := old.Fitness

			// Create a candidate by cloning and then mutating.
			cand := CloneIndividual(old)
			Mutate(cand, width, height, curMutation, colorOnly)
			candFit := LetterFitness(cand, target)

			// Keep mutation only if it improves fitness.
			if candFit > oldFit {
				cand.Fitness = candFit
				pop[i] = cand
				globalFit += candFit - oldFit
			}
		}

		// Compute average fitness for logging.
		avg := globalFit / float64(len(pop))

		// Here 'fit' uses global fitness (sum of all individuals)
		// for comparing runs inside this Evolve call.
		fit := globalFit
		pos := -1
		for i := 0; i < TopK; i++ {
			if fit > bestFits[i] {
				pos = i
				break
			}
		}
		// If this generation is in the top K, insert it into the arrays.
		if pos != -1 {
			for j := TopK - 1; j > pos; j-- {
				bestFits[j] = bestFits[j-1]
				bestGens[j] = bestGens[j-1]
				bestPops[j] = bestPops[j-1]
			}
			bestFits[pos] = fit
			bestGens[pos] = gen
			bestPops[pos] = ClonePopulation(pop)
		}

		// Log to file and print to console.
		fmt.Fprintf(logFile, "Generation %d best %.0f avg %.0f\n", gen, globalFit, avg)
		fmt.Printf("Generation %d best %.0f avg %.0f\n", gen, globalFit, avg)

		// Every 50 generations, save a snapshot image for visualization.
		if gen%50 == 0 {
			img := RenderPopulation(pop, width, height)
			prefix := strings.TrimSuffix(logSource, ".txt")
			snapPath := fmt.Sprintf("%s_gen_%04d.jpg", prefix, gen)
			_ = SavePNG(img, snapPath)
		}
	}

	// After evolution, render and save the TopK best populations.
	prefix := strings.TrimSuffix(logSource, ".txt")
	for i := 0; i < TopK; i++ {
		if bestPops[i] == nil {
			continue
		}
		img := RenderPopulation(bestPops[i], width, height)
		path := fmt.Sprintf("%s_best%d_gen_%04d.jpg", prefix, i+1, bestGens[i])
		_ = SavePNG(img, path)
	}

	return pop
}

// readLog parses one log file produced by Evolve.
// It returns a RunData with generation indices, best fitness, and average fitness.
func readLog(path string) (*RunData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data := &RunData{}
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()

		var gen int
		var best, avg float64

		// We expect lines in the format:
		// "Generation <gen> best <best> avg <avg>"
		n, err := fmt.Sscanf(line, "Generation %d best %f avg %f", &gen, &best, &avg)
		if err != nil || n != 3 {
			continue
		}

		data.Gen = append(data.Gen, float64(gen))
		data.Best = append(data.Best, best)
		data.Avg = append(data.Avg, avg)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return data, nil
}

// PlotRun draws one plot for a single run using RunData.
// X-axis shows fitness, Y-axis shows generation index.
// Two lines are plotted: best fitness and average fitness.
func PlotRun(data *RunData, title, outPath string) error {
	p := plot.New()
	p.Title.Text = title

	// Axis labels are set as requested:
	// X-axis is fitness, Y-axis is generation.
	p.X.Label.Text = "Fitness"
	p.Y.Label.Text = "Generation"

	bestPts := make(plotter.XYs, len(data.Gen))
	avgPts := make(plotter.XYs, len(data.Gen))

	for i := range data.Gen {
		// For each point, X is fitness, Y is generation index.
		bestPts[i].X = data.Best[i]
		bestPts[i].Y = data.Gen[i]

		avgPts[i].X = data.Avg[i]
		avgPts[i].Y = data.Gen[i]
	}

	bestLine, err := plotter.NewLine(bestPts)
	if err != nil {
		return err
	}
	avgLine, err := plotter.NewLine(avgPts)
	if err != nil {
		return err
	}

	p.Add(bestLine, avgLine)
	p.Legend.Add("best", bestLine)
	p.Legend.Add("avg", avgLine)

	p.Legend.Top = true
	p.Legend.Left = true

	// Save the plot to a PNG file with fixed size.
	if err := p.Save(6*vg.Inch, 4*vg.Inch, outPath); err != nil {
		return err
	}
	return nil
}

// CreatePlots loads log files for all images and all runs,
// then creates and saves fitness plots (with and without filter).
func CreatePlots() {
	// Ensure plot output directories exist.
	if err := os.MkdirAll("plots_no_filter", 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll("plots_with_filter", 0o755); err != nil {
		log.Fatal(err)
	}

	// For each image and each run, read log, then plot.
	for img := 1; img <= len(inputs); img++ {
		for run := 1; run <= NumRuns; run++ {

			// Plot for the run without filter.
			logPath := fmt.Sprintf(
				"outputs/outputs_no_filter/outputs_%d/output_%d_%d_logs_without_filter.txt",
				img, img, run,
			)

			data, err := readLog(logPath)
			if err != nil {
				log.Fatalf("cannot read %s: %v", logPath, err)
			}

			outPlot := fmt.Sprintf(
				"plots_no_filter/image_%d_run_%d.png",
				img, run,
			)
			title := fmt.Sprintf("Image %d, run %d (no filter)", img, run)

			if err := PlotRun(data, title, outPlot); err != nil {
				log.Fatalf("cannot plot %s: %v", outPlot, err)
			}
			fmt.Println("saved plot:", outPlot)

			// Plot for the run with filter.
			logPathF := fmt.Sprintf(
				"outputs/outputs_filter/outputs_%d/output_%d_%d_logs_with_filter.txt",
				img, img, run,
			)

			dataF, err := readLog(logPathF)
			if err != nil {
				log.Fatalf("cannot read %s: %v", logPathF, err)
			}

			outPlotF := fmt.Sprintf(
				"plots_with_filter/image_%d_run_%d.png",
				img, run,
			)
			titleF := fmt.Sprintf("Image %d, run %d (with filter)", img, run)

			if err := PlotRun(dataF, titleF, outPlotF); err != nil {
				log.Fatalf("cannot plot %s: %v", outPlotF, err)
			}
			fmt.Println("saved plot:", outPlotF)
		}
	}
}

// GenerateImages runs the evolutionary algorithm for all input images.
// For each image, it:
//  1. Runs the algorithm on the original image (no filter).
//  2. Creates a filtered version (blur + posterize + edges).
//  3. Runs the algorithm again on the filtered image.
//
// For each run, it saves logs and final images.
func GenerateImages() {
	for i, input := range inputs {
		im, err := LoadJPEG(input)
		if err != nil {
			panic(err)
		}

		imgIdx := i + 1

		noDir := fmt.Sprintf("outputs/outputs_no_filter/outputs_%d", imgIdx)
		withDir := fmt.Sprintf("outputs/outputs_filter/outputs_%d", imgIdx)
		if err := os.MkdirAll(noDir, 0o755); err != nil {
			panic(err)
		}
		if err := os.MkdirAll(withDir, 0o755); err != nil {
			panic(err)
		}

		b := im.Bounds()
		w, h := b.Dx(), b.Dy()

		// First, runs without any filter applied to target.
		for run := 1; run <= NumRuns; run++ {
			logNoFilter := fmt.Sprintf(
				"outputs/outputs_no_filter/outputs_%d/output_%d_%d_logs_without_filter.txt",
				imgIdx, imgIdx, run,
			)

			popLetters := Evolve(im, Generations, logNoFilter)

			outPath := fmt.Sprintf(
				"outputs/outputs_no_filter/outputs_%d/output_%d_%d_no_filter.jpg",
				imgIdx, imgIdx, run,
			)
			img := RenderPopulation(popLetters, w, h)
			if err := SavePNG(img, outPath); err != nil {
				panic(err)
			}
		}

		// Prepare the filtered version of the target image.
		blurred := Blur(im, 2)
		posterized := Posterize(blurred, PosterizeLevels)
		withEdges := DrawEdges(posterized, EdgeThreshold)

		// Save the filtered target for reference.
		targetPath := fmt.Sprintf(
			"outputs/outputs_filter/outputs_%d/output_%d_target_with_filter.jpg",
			imgIdx, imgIdx,
		)
		if err := SavePNG(withEdges, targetPath); err != nil {
			panic(err)
		}

		b2 := withEdges.Bounds()
		w2, h2 := b2.Dx(), b2.Dy()

		// Then, runs using the filtered image as target.
		for run := 1; run <= NumRuns; run++ {
			logWithFilter := fmt.Sprintf(
				"outputs/outputs_filter/outputs_%d/output_%d_%d_logs_with_filter.txt",
				imgIdx, imgIdx, run,
			)

			popLetters := Evolve(withEdges, Generations, logWithFilter)

			outPath := fmt.Sprintf(
				"outputs/outputs_filter/outputs_%d/output_%d_%d_filter.jpg",
				imgIdx, imgIdx, run,
			)
			img := RenderPopulation(popLetters, w2, h2)
			if err := SavePNG(img, outPath); err != nil {
				panic(err)
			}
		}
	}
}

// GenerateOnePicture is a helper that runs the algorithm only for the first
// input image and only once. This is useful for quick testing.
func GenerateOnePicture() {
	im, err := LoadJPEG(inputs[0])
	if err != nil {
		panic(err)
	}

	imgIdx := 1

	noDir := fmt.Sprintf("outputs/outputs_no_filter/outputs_%d", imgIdx)
	withDir := fmt.Sprintf("outputs/outputs_filter/outputs_%d", imgIdx)
	if err := os.MkdirAll(noDir, 0o755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll(withDir, 0o755); err != nil {
		panic(err)
	}

	b := im.Bounds()
	w, h := b.Dx(), b.Dy()

	// Here we use run index -1 only to distinguish this quick run
	// from the standard ones (1..NumRuns).
	logNoFilter := fmt.Sprintf(
		"outputs/outputs_no_filter/outputs_%d/output_%d_%d_logs_without_filter.txt",
		imgIdx, imgIdx, -1,
	)

	popLetters := Evolve(im, Generations, logNoFilter)

	outPath := fmt.Sprintf(
		"outputs/outputs_no_filter/outputs_%d/output_%d_%d_no_filter.jpg",
		imgIdx, imgIdx, -1,
	)
	img := RenderPopulation(popLetters, w, h)
	if err := SavePNG(img, outPath); err != nil {
		panic(err)
	}
}

// main is the entry point of the program.
// By default, it runs the full pipeline:
//  1. Run evolutionary algorithm for all images (with and without filter).
//  2. Create plots of fitness over generations for each run.
//
// You can comment/uncomment the calls below to only run a specific part.
func main() {
	//GenerateOnePicture() // Uncomment for a quick single-run test.
	GenerateImages()
	CreatePlots()
}
