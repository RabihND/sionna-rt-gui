CC = gcc

.SILENT:

# sudo apt install libcjson-dev
read_cir_batch: read_cir_batch.c
	$(CC) -o read_cir_batch read_cir_batch.c -lm -lcjson

clean:
	rm read_cir_batch

test: read_cir_batch
	./read_cir_batch cir_export.json
