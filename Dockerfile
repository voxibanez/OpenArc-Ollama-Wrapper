FROM golang:1.24-bookworm AS build

WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/ollama-openarc ./cmd/ollama-openarc

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/ollama-openarc /ollama-openarc
EXPOSE 11434
ENTRYPOINT ["/ollama-openarc"]
