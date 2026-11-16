FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# SocketCAN utilities (canplayer, candump, cansend, canbusload) and network interface tools (ip, modprobe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    can-utils \
    iproute2 \
    kmod \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/can-player/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
