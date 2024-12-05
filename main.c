#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define IMAGE_SIZE (28 * 28)   // MNIST goruntuleri icin 754 pixel boyutunda
#define MAX_LINE_LENGTH 10000 // csv icin gerekli
#define TRAIN_RATIO 0.8     // Egitim kumesi orani
#define DATASET_SAMPLE_COUNT 43000 // 42000 data var fakat daha fazla yer açmakta fayda var
#define MAX_SAMPLE_COUNT 9000 // anlik islevi yok sonradan ekleneb'l'r

#define BATCH_SIZE 256 // SGD ve ADAM icin ortak batch size boyutu
#define EPOCHS 20
#define LEARNING_RATE_GD 0.01
#define LEARNING_RATE_SGD 0.001
#define LEARNING_RATE_ADAM 0.001
#define WEIGHT_RANGE 0.001

// Adam
#define BETA1 0.9
#define BETA2  0.999
#define EPSILON 1e-8

float train_loss_gd[EPOCHS + 1], train_accuracy_gd[EPOCHS + 1];
float train_loss_sgd[EPOCHS + 1], train_accuracy_sgd[EPOCHS + 1];
float train_loss_adam[EPOCHS + 1], train_accuracy_adam[EPOCHS + 1];

float train_time_gd[EPOCHS + 1];
float train_time_sgd[EPOCHS + 1];
float train_time_adam[EPOCHS + 1];

// Hazir kod - Zaman olcumu icin gerekli
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void print_array(const char* name, float* arr, int size, FILE* file) {
    fprintf(file, "%s = [", name);
    for (int i = 0; i < size; i++) {
        fprintf(file, "%.5f", arr[i]);
        if (i < size - 1) fprintf(file, ", ");
    }
    fprintf(file, "]\n");
}

float tanh_activation(float x) {
    return tanh(x);
}

void read_csv(const char *file_path, int **labels, unsigned char ***images, int *num_samples) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("File not found!: %s\n", file_path);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;

    *labels = (int *)malloc(sizeof(int) * DATASET_SAMPLE_COUNT);
    *images = (unsigned char **)malloc(sizeof(unsigned char *) * DATASET_SAMPLE_COUNT);

    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        if (sample_count >= MAX_SAMPLE_COUNT) break;
        char *token;
        int pixel_index = 0;

        token = strtok(line, ",");
        int label = atoi(token);

        if (label == 0 || label == 1) {
            (*labels)[sample_count] = label;
            (*images)[sample_count] = (unsigned char *)malloc(sizeof(unsigned char) * IMAGE_SIZE);

            while ((token = strtok(NULL, ",")) != NULL) {
                (*images)[sample_count][pixel_index] = (unsigned char)atoi(token);
                pixel_index++;
            }

            sample_count++;
        }
    }

    *num_samples = sample_count;
    fclose(file);
}

void normalize_images(unsigned char **images, float ***normalized_images, int num_samples) {
    *normalized_images = (float **)malloc(sizeof(float *) * num_samples);

    for (int i = 0; i < num_samples; i++) {
        (*normalized_images)[i] = (float *)malloc(sizeof(float) * IMAGE_SIZE);
        for (int j = 0; j < IMAGE_SIZE; j++) {
            (*normalized_images)[i][j] = images[i][j] / 255.0f;
        }
    }
}

void split(float **images, int *labels, float ***train_images, int **train_labels,
                float ***test_images, int **test_labels, int num_samples) {

    int train_count = (int)(num_samples * TRAIN_RATIO);
    int test_count = num_samples - train_count;

    *train_images = (float **)malloc(sizeof(float *) * train_count);
    *train_labels = (int *)malloc(sizeof(int) * train_count);

    *test_images = (float **)malloc(sizeof(float *) * test_count);
    *test_labels = (int *)malloc(sizeof(int) * test_count);

    for (int i = 0; i < train_count; i++) {
        (*train_images)[i] = images[i];
        (*train_labels)[i] = labels[i];
    }

    for (int i = 0; i < test_count; i++) {
        (*test_images)[i] = images[train_count + i];
        (*test_labels)[i] = labels[train_count + i];
    }
}


void initialize_weights(float *weights, int input_size) {
    for (int i = 0; i < input_size; i++) {
         weights[i] = (((float)rand() / RAND_MAX) * (WEIGHT_RANGE * 2)) - WEIGHT_RANGE;
    }
}

float forward(float *weights, float *input, int size) {
    float dot_product = 0.0f;
    for (int i = 0; i < size - 1; i++) {
        dot_product += weights[i] * input[i];
    }
    dot_product += weights[size - 1]; // bias için sakın silme
    return tanh_activation(dot_product);
}

// Mean Squared Error hesabi
float calculate_mse(float prediction, int target) {
    return (prediction - target) * (prediction - target);
}

void update_weights_gd(float *weights, float **inputs, int *labels, float learning_rate, int num_samples, int size) {
    float *gradients = (float *)calloc(size, sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        float prediction = forward(weights, inputs[i], size);
        float error = prediction - labels[i];
        float derivative = 1.0f - (prediction * prediction); // tanh türevi

        for (int j = 0; j < size - 1; j++) {
            gradients[j] += error * derivative * inputs[i][j];
        }
        gradients[size - 1] += error * derivative; // bias icin gradyan
    }

    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * gradients[i] / num_samples;
    }

    free(gradients);
}

void update_weights_sgd(float *weights, float *input, int label, float learning_rate, int size) {
    float prediction = forward(weights, input, size);
    float error = prediction - label;

    for (int i = 0; i < size - 1; i++) {
        weights[i] -= learning_rate * error * (1.0f - prediction * prediction) * input[i];
    }
    weights[size - 1] -= learning_rate * error * (1.0f - prediction * prediction);
}


void update_weights_adam(float *weights, int *labels, int index, float **images, int size, float *first_moment, float *second_moment , int iter, float learning_rate) {

 // Gradyan hesabi
    float prediction = forward(weights, images[index], size);
    float *gradients = (float *)calloc(size, sizeof(float));
    float error = prediction - labels[index];
    float derivative = 1.0f - (prediction * prediction); //tanh türev

    for (int j = 0; j < size - 1; j++) {
        gradients[j] = error * derivative * images[index][j];
    }
    gradients[size - 1] = error * derivative;

    for (int i = 0; i < size; i++) {
        first_moment [i] = BETA1 * first_moment [i] + (1 - BETA1) * gradients[i];
        second_moment [i] = BETA2 * second_moment [i] + (1 - BETA2) * gradients[i] * gradients[i];

        float corrected_first_moment = first_moment [i] / (1 - powf(BETA1, iter));
        float corrected_second_moment  = second_moment[i] / (1 - powf(BETA2, iter));

        weights[i] -= learning_rate * corrected_first_moment / (sqrtf(corrected_second_moment ) + EPSILON);
    }

    free(gradients);
}

void train_gd(float **images, int *labels, int num_samples, float *weights, int size) {
    float final_accuracy = 0;
    float final_loss = 0;

    printf("Training with Gradient Descent...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_start_time = get_time();
        float total_loss = 0;
        int correct_predictions = 0;

        // forward
        for (int i = 0; i < num_samples; i++) {
            float prediction = forward(weights, images[i], size);
            total_loss += calculate_mse(prediction, labels[i]);
            if ((prediction >= 0.5 && labels[i] == 1) || (prediction < 0.5 && labels[i] == 0)) {
                correct_predictions++;
            }
        }

        // backward
        update_weights_gd(weights, images, labels, LEARNING_RATE_GD, num_samples, size);

        float avg_loss = total_loss / num_samples;
        float accuracy = (float)correct_predictions / num_samples * 100;
        final_accuracy = accuracy;
        final_loss = avg_loss;

        double epoch_end_time = get_time();

        // Python dosyası icin veri yazimi
        train_loss_gd[epoch + 1] = avg_loss;
        train_accuracy_gd[epoch + 1] = accuracy;
        train_time_gd[epoch + 1] = train_time_gd[epoch] + epoch_end_time - epoch_start_time;


        printf("Epoch %d | Loss: %.5f | Accuracy: %.3f%% | Time: %.4f seconds\n",
               epoch + 1, avg_loss, accuracy, epoch_end_time - epoch_start_time);

    }
    printf("\nGradient Descent Training Accuracy: %.3f%%\n", final_accuracy);
    printf("Gradient Descent Final Training Loss: %.5f\n", final_loss);

}

void train_sgd(float **images, int *labels, int num_samples, float *weights, int size) {
    float final_accuracy = 0;
    float final_loss = 0;


    printf("Training with Stochastic Gradient Descent...\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_start_time = get_time();
        float total_loss = 0.0f;
        int correct_predictions = 0;


        for (int step = 0; step < BATCH_SIZE; step++) {
            int index = rand()%num_samples;

            float prediction = forward(weights, images[index], size);
            total_loss += calculate_mse(prediction, labels[index]);

            int prediction_class = (prediction >= 0.5f) ? 1 : 0;
            if (prediction_class == labels[index]) {
                correct_predictions++;
            }
            update_weights_sgd(weights, images[index], labels[index], LEARNING_RATE_SGD, size);
        }

        float avg_loss = total_loss / BATCH_SIZE;
        float accuracy = (float)correct_predictions / BATCH_SIZE * 100.0f;
        final_accuracy = accuracy;
        final_loss = avg_loss;

        double epoch_end_time = get_time();

        // python kodları
        train_loss_sgd[epoch + 1] = avg_loss;
        train_accuracy_sgd[epoch + 1] = accuracy;
        train_time_sgd[epoch + 1] = train_time_sgd[epoch] + epoch_end_time - epoch_start_time;
        printf("Epoch %d | Loss: %.5f | Accuracy: %.3f%% | Time: %.4f seconds\n",
               epoch + 1, avg_loss, accuracy, epoch_end_time - epoch_start_time);

    }
    printf("\nStochastic Gradient Descent Training Accuracy: %.3f%%\n", final_accuracy);
    printf("Stochastic Gradient Descent Final Training Loss: %.5f\n", final_loss);

}


void train_adam(float **images, int *labels, int num_samples, float *weights, int size) {
    float *first_moment = (float *)calloc(size, sizeof(float));
    float *second_moment = (float *)calloc(size, sizeof(float));
    int iteration = 0;

    float final_accuracy = 0;
    float final_loss = 0;

    printf("Training with ADAM...\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_start_time = get_time();
        float total_loss = 0.0f;
        int correct_predictions = 0;

        for (int step = 0; step < BATCH_SIZE; step++) {
            int index = rand()%num_samples;

            // forward
            float prediction = forward(weights, images[index], size);
            total_loss += calculate_mse(prediction, labels[index]);

             prediction = (prediction >= 0.5) ? 1 : 0;
            if (prediction == labels[index]) {
                correct_predictions++;
            }


            // backward
            iteration++;
            update_weights_adam(weights, labels, index, images, size, first_moment, second_moment, iteration, LEARNING_RATE_ADAM);
        }

        float avg_loss = total_loss / BATCH_SIZE;
        float accuracy = (float)correct_predictions / BATCH_SIZE * 100.0f;
        final_accuracy = accuracy;
        final_loss = avg_loss;

        double epoch_end_time = get_time();

        // Python çıktısı için
        train_loss_adam[epoch + 1] = avg_loss;
        train_accuracy_adam[epoch + 1] = accuracy;
        train_time_adam[epoch + 1] = train_time_adam[epoch] + epoch_end_time - epoch_start_time;

        printf("Epoch %d | Loss: %.5f | Accuracy: %.3f%% | Time: %.4f seconds\n",
               epoch + 1, avg_loss, accuracy, epoch_end_time - epoch_start_time);

    }

    printf("\nADAM Training Accuracy: %.3f%%\n", final_accuracy);
    printf("ADAM Final Training Loss: %.5f\n", final_loss);

    free(first_moment);
    free(second_moment);
}

int main() {
    srand(time(NULL));

    const char *file_path = "data/train.csv";
    int *labels = NULL;
    unsigned char **images = NULL;
    int num_samples = 0;

    read_csv(file_path, &labels, &images, &num_samples);

    float **normalized_images = NULL;
    normalize_images(images, &normalized_images, num_samples);

    float **train_images = NULL, **test_images = NULL;
    int *train_labels = NULL, *test_labels = NULL;
    split(normalized_images, labels, &train_images, &train_labels, &test_images, &test_labels, num_samples);

    int input_size = IMAGE_SIZE + 1;

    // Ağırlıkları bir kez initialize et
    float *weights = (float *)malloc(sizeof(float) * input_size);
    initialize_weights(weights, input_size);

    // GD, SGD ve Adam için ağırlık kopyaları
    float *weights_gd = (float *)malloc(sizeof(float) * input_size);
    memcpy(weights_gd, weights, sizeof(float) * input_size);

    float *weights_sgd = (float *)malloc(sizeof(float) * input_size);
    memcpy(weights_sgd, weights, sizeof(float) * input_size);

    float *weights_adam = (float *)malloc(sizeof(float) * input_size);
    memcpy(weights_adam, weights, sizeof(float) * input_size);

    double start_time, end_time;

    // Başlangıç loss değerlerini hesapla ve kaydet
    float initial_loss = 0.0f;
    for (int i = 0; i < (int)(num_samples * TRAIN_RATIO); i++) {
        float prediction = forward(weights, train_images[i], input_size);
        initial_loss += calculate_mse(prediction, train_labels[i]);
    }

    initial_loss /= (int)(num_samples * TRAIN_RATIO);
    printf("Initial loss: %f\n\n", initial_loss);

    // GD Egitimi
    start_time = get_time();
    train_gd(train_images, train_labels, (int)(num_samples * TRAIN_RATIO), weights_gd, input_size);
    end_time = get_time();
    printf("Gradient Descent total time: %.4f seconds\n\n", end_time - start_time);

    // SGD Egitimi
    start_time = get_time();
    train_sgd(train_images, train_labels, (int)(num_samples * TRAIN_RATIO), weights_sgd, input_size);
    end_time = get_time();
    printf("Stochastic Gradient Descent total time: %.4f seconds\n\n", end_time - start_time);

    // Adam Egitimi
    start_time = get_time();
    train_adam(train_images, train_labels, (int)(num_samples * TRAIN_RATIO), weights_adam, input_size);
    end_time = get_time();
    printf("ADAM total time: %.4f seconds\n\n", end_time - start_time);


    // Test Asamasi
    int test_count = num_samples - (int)(num_samples * TRAIN_RATIO);
    float test_loss_gd[test_count / 100 + 1], test_accuracy_gd[test_count / 100 + 1];
    float test_loss_sgd[test_count / 100 + 1], test_accuracy_sgd[test_count / 100 + 1];
    float test_loss_adam[test_count / 100 + 1], test_accuracy_adam[test_count / 100 + 1];

    float test_times[test_count / 100 + 1];

    float *test_weights[] = {weights_gd, weights_sgd, weights_adam};
    float *test_loss[] = {test_loss_gd, test_loss_sgd, test_loss_adam};
    float *test_accuracy[] = {test_accuracy_gd, test_accuracy_sgd, test_accuracy_adam};
    char *algorithm_names[] = {"GD", "SGD", "ADAM"};

    for (int alg = 0; alg < 3; alg++) {
        float total_test_time = 0;
        int test_output_count = 0;
        int correct_predictions = 0;
        float total_loss = 0.0f;

        for (int i = 0; i < test_count; i++) {
            start_time = get_time();
            float prediction = forward(test_weights[alg], test_images[i], input_size);
            prediction = (prediction >= 0.5) ? 1 : 0;

            if (prediction == test_labels[i]) {
                correct_predictions++;
            }

            total_loss += calculate_mse(prediction, test_labels[i]);
            end_time = get_time();
            total_test_time += end_time - start_time;
            if (i % 100 == 0) {
                float accuracy = (float)correct_predictions / (i + 1) * 100.0f;
                float loss = (float)total_loss / (i + 1);

                test_loss[alg][test_output_count] = loss;

                test_accuracy[alg][test_output_count] = accuracy;

                test_times[test_output_count] = total_test_time;

                test_output_count++;
                printf("%s TEST, Step: %d, Total Loss: %.5f, Total Accuracy: %.2f%%\n", algorithm_names[alg],i, loss, accuracy);
            }
        }

        float accuracy = (float)correct_predictions / test_count * 100.0f;
        float avg_loss = total_loss / test_count;

        printf("%s Test Accuracy: %.2f%%\n", algorithm_names[alg], accuracy);
        printf("%s Test Loss: %.5f\n\n", algorithm_names[alg], avg_loss);
    }

    // Grafik icin kolay cikti alim kismi

    train_loss_gd[0] = initial_loss;
    train_loss_sgd[0] = initial_loss;
    train_loss_adam[0] = initial_loss;

    FILE* file = fopen("results.txt", "w");
    if (file == NULL) {
        printf("File not found!\n");
        return 0;
    }

    fprintf(file, "epochs = [");
    for (int i = 0; i < EPOCHS + 1; i++) {
        fprintf(file, "%d", i);
        if (i < EPOCHS) fprintf(file, ", ");
    }
    fprintf(file, "]\n");

    fprintf(file, "test_steps = [");
    for (int i = 0; i < test_count / 100 + 1; i++) {
        fprintf(file, "%d", i * 100);
        if (i < test_count / 100) fprintf(file, ", ");
    }
    fprintf(file, "]\n");

    // Algoritmaların loss ve accuracy değerlerini yazdırma
    print_array("train_loss_gd", train_loss_gd, EPOCHS + 1, file);
    print_array("train_loss_sgd", train_loss_sgd, EPOCHS + 1, file);
    print_array("train_loss_adam", train_loss_adam, EPOCHS + 1, file);


    print_array("train_accuracy_gd", train_accuracy_gd, EPOCHS + 1, file);
    print_array("train_accuracy_sgd", train_accuracy_sgd, EPOCHS + 1, file);
    print_array("train_accuracy_adam", train_accuracy_adam, EPOCHS + 1, file);


    print_array("train_time_gd", train_time_gd, EPOCHS + 1, file);
    print_array("train_time_sgd", train_time_sgd, EPOCHS + 1, file);
    print_array("train_time_adam", train_time_adam, EPOCHS + 1, file);

    print_array("test_loss_gd", test_loss_gd, test_count / 100 + 1, file);
    print_array("test_loss_sgd", test_loss_sgd, test_count / 100 + 1, file);
    print_array("test_loss_adam", test_loss_adam, test_count / 100 + 1, file);

    print_array("test_accuracy_gd", test_accuracy_gd, test_count / 100 + 1, file);
    print_array("test_accuracy_sgd", test_accuracy_sgd, test_count / 100 + 1, file);
    print_array("test_accuracy_adam", test_accuracy_adam, test_count / 100 + 1, file);

    print_array("test_times", test_times, test_count / 100 + 1, file);

    fprintf(file, "weight_range = %0.3f\n", WEIGHT_RANGE);
    fprintf(file, "batch_size = %d", BATCH_SIZE);

    // Dosyayı kapatma
    fclose(file);

    printf("Results are saved to 'results.txt' file.\n");

    free(labels);
    for (int i = 0; i < num_samples; i++) {
        free(images[i]);
        free(normalized_images[i]);
    }

    free(images);
    free(normalized_images);
    free(train_labels);
    free(test_labels);
    free(train_images);
    free(test_images);
    free(weights);
    free(weights_gd);
    free(weights_sgd);
    free(weights_adam);

    return 0;
}

