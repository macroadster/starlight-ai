// trin_forward runs BalancedStarlightDetector from models/starlight.gguf
// through github.com/macroadster/trin/pkg/starlight.
//
// Stdin is little-endian float32 arrays, each prefixed by a uint32 count,
// in this order: pixel, meta, alpha, lsb, palette, palette_lsb, format, content.
// Stdout is one JSON object: stego logit, method logits (5), bit-order logits (3).
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	trinstar "github.com/macroadster/trin/pkg/starlight"
)

const spatial = 256

func expectedCounts() []int {
	plane := spatial * spatial
	return []int{
		3 * plane, // pixel CHW
		2048,      // meta
		2 * plane, // alpha
		3 * plane, // lsb CHW
		768,       // palette
		plane,     // palette LSB
		6,         // format
		6,         // content
	}
}

func readArray(r io.Reader) ([]float32, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n > 2_000_000 {
		return nil, fmt.Errorf("array length %d is too large", n)
	}
	buf := make([]float32, n)
	if n == 0 {
		return buf, nil
	}
	if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func main() {
	gguf := flag.String("gguf", "", "path to models/starlight.gguf")
	flag.Parse()
	if *gguf == "" {
		fmt.Fprintln(os.Stderr, "trin_forward: --gguf is required")
		os.Exit(2)
	}

	want := expectedCounts()
	got := make([][]float32, len(want))
	for i, n := range want {
		arr, err := readArray(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "trin_forward: read array %d: %v\n", i, err)
			os.Exit(1)
		}
		if len(arr) != n {
			fmt.Fprintf(os.Stderr, "trin_forward: array %d length %d, want %d\n", i, len(arr), n)
			os.Exit(1)
		}
		got[i] = arr
	}

	sess, err := trinstar.Open(*gguf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "trin_forward: open: %v\n", err)
		os.Exit(1)
	}
	defer sess.Close()

	out, err := sess.Forward(trinstar.Inputs{
		Pixel:           got[0],
		Meta:            got[1],
		Alpha:           got[2],
		LSB:             got[3],
		Palette:         got[4],
		PaletteLSB:      got[5],
		FormatFeatures:  got[6],
		ContentFeatures: got[7],
		H:               spatial,
		W:               spatial,
		MetaWeight:      0.3,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "trin_forward: forward: %v\n", err)
		os.Exit(1)
	}
	if out == nil || len(out.Stego) < 1 || len(out.Method) < 5 || len(out.BitOrder) < 3 {
		fmt.Fprintln(os.Stderr, "trin_forward: short head output")
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(map[string]any{
		"stego_logit": out.Stego[0],
		"method":      out.Method[:5],
		"bit_order":   out.BitOrder[:3],
	}); err != nil {
		fmt.Fprintf(os.Stderr, "trin_forward: write: %v\n", err)
		os.Exit(1)
	}
}
