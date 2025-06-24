# Dockerfile for mdbook with mdbook-katex
FROM rust:alpine AS builder

# Install dependencies
RUN apk add --no-cache \
    musl-dev \
    gcc \
    && cargo install mdbook mdbook-katex

# Set working directory
WORKDIR /app

# Copy book files
COPY book.toml src theme /app/

# Build the book
RUN mdbook build

# Use a lightweight image for runtime
FROM alpine:latest

# Copy built files from builder
COPY --from=builder /app/book /app/book

# Expose port for serving
EXPOSE 3000

# Serve the book
CMD ["mdbook", "serve", "--open", "--hostname", "0.0.0.0", "--port", "3000"]