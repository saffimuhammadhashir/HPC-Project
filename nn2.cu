#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE    784
#define HIDDEN_SIZE   128
#define OUTPUT_SIZE   10
#define LEARNING_RATE 0.01
#define EPOCHS        5

// Timer
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate/free host matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        mat[i] = (double*)malloc(cols * sizeof(double));
    return mat;
}
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}

// MNIST loaders
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror(filename); exit(1); }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, 1, 1, file) != 1) {
                fprintf(stderr, "Error reading image\n");
                exit(1);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}
double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror(filename); exit(1); }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, 1, 1, file) != 1) {
            fprintf(stderr, "Error reading label\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++)
            labels[i][j] = (j == label) ? 1.0 : 0.0;
    }
    fclose(file);
    return labels;
}

// CUDA kernels

__global__ void kernel_forward_hidden(
    const double* __restrict__ d_W1,
    const double* __restrict__ d_b1,
    const double* __restrict__ d_input,
          double*       d_hidden
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += d_W1[i*INPUT_SIZE + j] * d_input[j];
        d_hidden[i] = (sum > 0.0) ? sum : 0.0;
    }
}

__global__ void kernel_forward_output(
    const double* __restrict__ d_W2,
    const double* __restrict__ d_b2,
    const double* __restrict__ d_hidden,
          double*       d_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        double sum = d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += d_W2[i*HIDDEN_SIZE + j] * d_hidden[j];
        d_output[i] = sum;
    }
}

__global__ void softmaxkernel(double* x, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

__global__ void kernel_compute_output_grad(
    const double* __restrict__ d_output,
    const double* __restrict__ d_target,
          double*       d_grad_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE)
        d_grad_output[i] = d_output[i] - d_target[i];
}

__global__ void kernel_compute_hidden_grad(
    const double* __restrict__ d_W2,
    const double* __restrict__ d_hidden,
    const double* __restrict__ d_grad_output,
          double*             d_grad_hidden
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            sum += d_W2[j*HIDDEN_SIZE + i] * d_grad_output[j];
        d_grad_hidden[i] = (d_hidden[i] > 0.0) ? sum : 0.0;
    }
}

__global__ void kernel_update_W2(
    double*       d_W2,
    const double* d_hidden,
    const double* d_grad_output
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
        int idx = i*HIDDEN_SIZE + j;
        d_W2[idx] -= LEARNING_RATE * d_grad_output[i] * d_hidden[j];
    }
}

__global__ void kernel_update_W1(
    double*       d_W1,
    const double* d_input,
    const double* d_grad_hidden
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < HIDDEN_SIZE && j < INPUT_SIZE) {
        int idx = i*INPUT_SIZE + j;
        d_W1[idx] -= LEARNING_RATE * d_grad_hidden[i] * d_input[j];
    }
}

__global__ void kernel_update_b2(
    double*       d_b2,
    const double* d_grad_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE)
        d_b2[i] -= LEARNING_RATE * d_grad_output[i];
}

__global__ void kernel_update_b1(
    double*       d_b1,
    const double* d_grad_hidden
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE)
        d_b1[i] -= LEARNING_RATE * d_grad_hidden[i];
}

// Network struct
typedef struct {
    double** W1; double** W2;
    double*   b1; double*   b2;
    double* d_W1; double* d_b1;
    double* d_W2; double* d_b2;
    double* d_input; double* d_hidden; double* d_output;
    double* d_target; double* d_grad_output; double* d_grad_hidden;
} NeuralNetwork;

// Create host network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand()/RAND_MAX)*0.01;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand()/RAND_MAX)*0.01;
    return net;
}

// Allocate device buffers
void allocDeviceNetwork(NeuralNetwork* net) {
    cudaMalloc(&net->d_W1, HIDDEN_SIZE*INPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&net->d_W2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&net->d_b2, OUTPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_input,      INPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_hidden,     HIDDEN_SIZE*sizeof(double));
    cudaMalloc(&net->d_output,     OUTPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_target,     OUTPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_grad_output, OUTPUT_SIZE*sizeof(double));
    cudaMalloc(&net->d_grad_hidden, HIDDEN_SIZE*sizeof(double));
}

// Upload host→device
void uploadDeviceNetwork(NeuralNetwork* net) {
    for (int i = 0; i < HIDDEN_SIZE; i++)
        cudaMemcpy(net->d_W1 + i*INPUT_SIZE, net->W1[i],
                   INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        cudaMemcpy(net->d_W2 + i*HIDDEN_SIZE, net->W2[i],
                   HIDDEN_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
}

// Training
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0; int correct = 0;
        for (int i = 0; i < numImages; i++) {
            double output[OUTPUT_SIZE];

            // forward
            cudaMemcpy(net->d_input, images[i], INPUT_SIZE*sizeof(double),
                       cudaMemcpyHostToDevice);

            int t = 256, b;
            b = (HIDDEN_SIZE + t - 1)/t;
            kernel_forward_hidden<<<b, t>>>(net->d_W1, net->d_b1,
                                            net->d_input, net->d_hidden);
            cudaDeviceSynchronize();

            b = (OUTPUT_SIZE + t - 1)/t;
            kernel_forward_output<<<b, t>>>(net->d_W2, net->d_b2,
                                            net->d_hidden, net->d_output);
            cudaDeviceSynchronize();

            softmaxkernel<<<1,1>>>(net->d_output, OUTPUT_SIZE);
            cudaDeviceSynchronize();


            // backward
            cudaMemcpy(net->d_target, labels[i], OUTPUT_SIZE*sizeof(double),
                       cudaMemcpyHostToDevice);

            b = (OUTPUT_SIZE + t - 1)/t;
            kernel_compute_output_grad<<<b, t>>>(net->d_output, net->d_target,
                                                 net->d_grad_output);
            cudaDeviceSynchronize();

            b = (HIDDEN_SIZE + t - 1)/t;
            kernel_compute_hidden_grad<<<b, t>>>(net->d_W2, net->d_hidden,
                                                 net->d_grad_output,
                                                 net->d_grad_hidden);
            cudaDeviceSynchronize();

            dim3 block2d(16,16);
            dim3 gridW2((HIDDEN_SIZE+15)/16, (OUTPUT_SIZE+15)/16);
            kernel_update_W2<<<gridW2, block2d>>>(net->d_W2, net->d_hidden,
                                                  net->d_grad_output);
            cudaDeviceSynchronize();

            dim3 gridW1((INPUT_SIZE+15)/16, (HIDDEN_SIZE+15)/16);
            kernel_update_W1<<<gridW1, block2d>>>(net->d_W1, net->d_input,
                                                  net->d_grad_hidden);
            cudaDeviceSynchronize();

            b = (OUTPUT_SIZE + t - 1)/t;
            kernel_update_b2<<<b, t>>>(net->d_b2, net->d_grad_output);
            cudaDeviceSynchronize();

            b = (HIDDEN_SIZE + t - 1)/t;
            kernel_update_b1<<<b, t>>>(net->d_b1, net->d_grad_hidden);
            cudaDeviceSynchronize();

            // copy out
            cudaMemcpy(output, net->d_output, OUTPUT_SIZE*sizeof(double),
            cudaMemcpyDeviceToHost);

            // compute loss/accuracy on CPU
            for (int k = 0; k < OUTPUT_SIZE; k++)
                loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
            epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    double output[OUTPUT_SIZE];
    for (int i = 0; i < numImages; i++) {
        cudaMemcpy(net->d_input, images[i], INPUT_SIZE*sizeof(double),
                   cudaMemcpyHostToDevice);
        int t=256, b;
        b = (HIDDEN_SIZE + t - 1)/t;
        kernel_forward_hidden<<<b,t>>>(net->d_W1, net->d_b1,
                                       net->d_input, net->d_hidden);
        cudaDeviceSynchronize();
        b = (OUTPUT_SIZE + t - 1)/t;
        kernel_forward_output<<<b,t>>>(net->d_W2, net->d_b2,
                                       net->d_hidden, net->d_output);
        cudaDeviceSynchronize();
        softmaxkernel<<<1,1>>>(net->d_output, OUTPUT_SIZE);
        cudaDeviceSynchronize();
        cudaMemcpy(output, net->d_output, OUTPUT_SIZE*sizeof(double),
                   cudaMemcpyDeviceToHost);
        int pred=0, act=0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][act]) act = j;
        }
        if (pred == act) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", 100.0*correct/numImages);
}

int main() {
    printf("MNIST Neural Network V2\n");

    // const char* basepath1 = "C:/Users/abgho/Documents/Uni Work/HPC/Project/data/";
    const char* basepath2 = "H:/Github Projects/HPC Project/data/";

    char train_images_path[256], train_labels_path[256];
    char test_images_path[256], test_labels_path[256];

    sprintf(train_images_path, "%strain-images.idx3-ubyte", basepath2);
    sprintf(train_labels_path, "%strain-labels.idx1-ubyte", basepath2);
    sprintf(test_images_path,  "%st10k-images.idx3-ubyte", basepath2);
    sprintf(test_labels_path,  "%st10k-labels.idx1-ubyte", basepath2);

    double** train_images = loadMNISTImages(train_images_path, 60000);
    double** train_labels = loadMNISTLabels(train_labels_path, 60000);
    double** test_images = loadMNISTImages(test_images_path, 10000);
    double** test_labels = loadMNISTLabels(test_labels_path, 10000);

    NeuralNetwork* net = createNetwork();
    allocDeviceNetwork(net);
    uploadDeviceNetwork(net);

    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images,  10000);
    freeMatrix(test_labels,  10000);
    
    return 0;
}
