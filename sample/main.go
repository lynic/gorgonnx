package main

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/owulveryck/gorgonnx"
	onnx "github.com/owulveryck/onnx/go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor/tensonnx"
)

func main() {
	b, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	model := new(onnx.ModelProto)
	err = model.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	g, err := gorgonnx.NewGraph(model.GetGraph())
	if err != nil {
		log.Fatal("Cannot decode ", err)
	}

	//fmt.Println(g.ToDot())
	// Open the tensorproto sample file

	b, err = ioutil.ReadFile(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	sampleTestData := new(onnx.TensorProto)
	err = sampleTestData.Unmarshal(b)
	if err != nil {
		log.Fatal(err)
	}
	t, err := tensonnx.NewTensor(sampleTestData)
	if err != nil {
		log.Fatal(err)
	}
	gorgonia.Let(g.ByName("Input3")[0], t)
	/*
		for _, n := range g.Inputs() {
			if n.Name() == "Input3" {
				gorgonia.Let(n, t)
			}
		}
	*/
	machine := gorgonia.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	for _, n := range g.AllNodes() {
		if len(n.Shape()) == 2 && n.Shape()[0] == 1 && n.Shape()[1] == 10 {
			log.Printf("%v: %v", n.Name(), n.Value())
		}
	}
}
