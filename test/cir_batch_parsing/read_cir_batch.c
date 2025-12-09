#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <cjson/cJSON.h>


typedef struct {
    int n_cirs;
    int n_taps;
    double duration_s;
} BatchInfo;


bool read_batch_info(const char *json_filename, BatchInfo *batch_info) {
    FILE *file = fopen(json_filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file %s\n", json_filename);
        return false;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *file_content = malloc(file_size + 1);
    fread(file_content, 1, file_size, file);
    file_content[file_size] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(file_content);
    free(file_content);  // TODO: does cJSON need this to stay live?
    if (!root) {
        fprintf(stderr, "Error: Failed to parse JSON file %s\n", json_filename);
        return false;
    }

    cJSON *batch = cJSON_GetObjectItem(root, "batch");
    if (!batch) {
        fprintf(stderr, "Error: Failed to get batch object\n");
        return false;
    }

    batch_info->n_cirs = cJSON_GetObjectItem(batch, "n_cirs")->valueint;
    batch_info->duration_s = cJSON_GetObjectItem(batch, "duration_s")->valuedouble;


    cJSON *paths = cJSON_GetObjectItem(root, "paths");
    if (!paths) {
        fprintf(stderr, "Error: Failed to get paths object\n");
        return false;
    }
    batch_info->n_taps = cJSON_GetObjectItem(paths, "num_taps")->valueint;

    return true;
}


bool read_cir_batch(const char *json_filename, BatchInfo batch_info) {
    // Derive bin filename from json filename
    char *bin_filename = malloc(strlen(json_filename) + 5);
    strcpy(bin_filename, json_filename);
    strcpy(bin_filename + strlen(json_filename) - strlen(".json"), ".bin");

    // Open bin file
    FILE *file = fopen(bin_filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file %s\n", bin_filename);
        return false;
    }
    free(bin_filename);

    // Read CIRs
    for (int i = 0; i < batch_info.n_cirs; i++) {
        // For each CIR, one float for the norm, then the CIR taps.
        float norm;
        fread(&norm, sizeof(float), 1, file);

        // (Real part, imaginary part) for each tap.
        float cir[batch_info.n_taps * 2];
        fread(cir, sizeof(float), batch_info.n_taps * 2, file);

        // Tap indices.
        uint16_t tap_indices[batch_info.n_taps];
        fread(tap_indices, sizeof(uint16_t), batch_info.n_taps, file);

        printf("--- CIR %d (norm = %f): [", i, norm);
        for (int j = 0; j < batch_info.n_taps; j++) {
            printf("%f + %fj, ", cir[j * 2], cir[j * 2 + 1]);
        }
        printf("]\n");
        printf("    Tap indices: [");
        for (int j = 0; j < batch_info.n_taps; j++) {
            printf("%d, ", tap_indices[j]);
        }
        printf("]\n");
    }

    // Make sure we read the whole file.
    if (fgetc(file) != EOF) {
        fprintf(stderr, "Error: Extra data at end of file.\n");
        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <cir_export.json>\n", argv[0]);
        return 1;
    }

    char *json_filename = argv[1];
    printf("[i] Reading CIR batch from %s\n", json_filename);

    BatchInfo batch_info;
    if (!read_batch_info(json_filename, &batch_info)) {
        fprintf(stderr, "Error: Failed to read batch info\n");
        return 1;
    }

    printf("[i] Found %d CIRs, duration = %.2f seconds\n", batch_info.n_cirs, batch_info.duration_s);

    if (!read_cir_batch(json_filename, batch_info)) {
        fprintf(stderr, "Error: Failed to read CIRs\n");
        return 1;
    }
}
