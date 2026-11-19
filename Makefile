.PHONY: test vet build

test:
	go test -race -v ./...

vet:
	go vet ./...

build:
	go build ./...
