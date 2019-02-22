Run Tests

The file testbackend.py will run the conv tests against onnxruntime or tensorflow backend.


mnist_test.go cotains the same tests to run against gorgonia, and  it didn't give out the  correct result.

Run tests against onnxruntiime:
```
docker pull elynn/onnxrt:latesst
docker run --rm -it -v $(pwd):/opt/test/ elynn//onnxrt:latest
cd /opt/test
python testbackend.py
```

