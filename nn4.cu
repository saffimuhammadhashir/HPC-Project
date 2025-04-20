#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INPUT_SIZE    784
#define HIDDEN_SIZE   256
#define OUTPUT_SIZE   10
#define LEARNING_RATE 0.001f
#define EPOCHS        3
#define BATCH_SIZE    64

// Timer
__host__ double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate/free host matrix
float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++)
        mat[i] = (float*)malloc(cols * sizeof(float));
    return mat;
}

void freeMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}

// MNIST loaders
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror(filename); exit(1); }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, 1, 1, file) != 1) {
                fprintf(stderr, "Error reading image\n");
                exit(1);
            }
            images[i][j] = pixel / 255.0f;
        }
    }
    fclose(file);
    return images;
}

float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror(filename); exit(1); }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, 1, 1, file) != 1) {
            fprintf(stderr, "Error reading label\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++)
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
    }
    fclose(file);
    return labels;
}

// CUDA kernels
__global__ void batch_relu_kernel(float* data, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

__global__ void batch_add_bias_kernel(float* data, const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        data[idx] += bias[row];
    }
}

__global__ void batch_softmax_kernel(float* data, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float max_val = -INFINITY;
        for (int i = 0; i < rows; i++) {
            float val = data[i * cols + col];
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            float val = expf(data[i * cols + col] - max_val);
            data[i * cols + col] = val;
            sum += val;
        }
        for (int i = 0; i < rows; i++) {
            data[i * cols + col] /= sum;
        }
    }
}

__global__ void compute_batch_grad_output(float* output, float* labels, float* grad_output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        grad_output[idx] = output[idx] - labels[idx];
    }
}

__global__ void row_sum_kernel(float* matrix, float* sum, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float s = 0.0f;
        for (int c = 0; c < cols; c++) {
            s += matrix[row * cols + c];
        }
        sum[row] = s;
    }
}

__global__ void relu_derivative_kernel(float* grad_hidden, float* hidden, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        grad_hidden[idx] *= (hidden[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void update_weights_kernel(float* weights, float* grad, int size, float lr, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        weights[i] -= lr * grad[i] / batch_size;
    }
}

// Neural network struct
typedef struct {
    float* d_batch_input;
    float* d_batch_labels;
    float* d_batch_hidden;
    float* d_batch_output;
    float* d_batch_grad_hidden;
    float* d_batch_grad_output;
    float* d_W1;
    float* d_b1;
    float* d_W2;
    float* d_b2;
    float* d_sum_grad_W1;
    float* d_sum_grad_W2;
    float* d_sum_grad_b1;
    float* d_sum_grad_b2;
    float** W1;
    float** W2;
    float* b1;
    float* b2;
    float* host_batch_input;
    float* host_batch_labels;
} NeuralNetwork;

// Create network with He initialization
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    // Host-side initialization with He initialization
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    srand(time(NULL));


    float a_W1 = sqrtf(6.0f / INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * a_W1;


    float a_W2 = sqrtf(6.0f / HIDDEN_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * a_W2;

    // Host-side batch memory
    net->host_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    net->host_batch_labels = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Device-side memory allocation
    cudaMalloc(&net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_batch_labels, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_batch_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_batch_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_sum_grad_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&net->d_sum_grad_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_sum_grad_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&net->d_sum_grad_b2, OUTPUT_SIZE * sizeof(float));

    return net;
}

// Upload host to device
void uploadDeviceNetwork(NeuralNetwork* net) {
    for (int i = 0; i < HIDDEN_SIZE; i++)
        cudaMemcpy(net->d_W1 + i * INPUT_SIZE, net->W1[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        cudaMemcpy(net->d_W2 + i * HIDDEN_SIZE, net->W2[i], HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
}


// Train function with proper batch processing
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t total_start = clock();

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    float* host_batch_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;

        for (int batch_start = 0; batch_start < numImages; batch_start += BATCH_SIZE) {
            int actual_batch = (batch_start + BATCH_SIZE < numImages) ? BATCH_SIZE : numImages - batch_start;

            // Load batch data
            for (int b = 0; b < actual_batch; b++) {
                for (int j = 0; j < INPUT_SIZE; j++)
                    net->host_batch_input[b * INPUT_SIZE + j] = images[batch_start + b][j];
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    net->host_batch_labels[b * OUTPUT_SIZE + j] = labels[batch_start + b][j];
            }


            cudaMemcpy(net->d_batch_input, net->host_batch_input, actual_batch * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(net->d_batch_labels, net->host_batch_labels, actual_batch * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass: hidden layer
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_SIZE, actual_batch, INPUT_SIZE, &alpha, net->d_W1, HIDDEN_SIZE, net->d_batch_input, INPUT_SIZE, &beta, net->d_batch_hidden, HIDDEN_SIZE);
        
            dim3 block(16, 16);
            dim3 grid((actual_batch + block.x - 1) / block.x, (HIDDEN_SIZE + block.y - 1) / block.y);
            batch_add_bias_kernel<<<grid, block>>>(net->d_batch_hidden, net->d_b1, HIDDEN_SIZE, actual_batch);
            batch_relu_kernel<<<grid, block>>>(net->d_batch_hidden, HIDDEN_SIZE, actual_batch);


            // Forward pass: output layer
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, actual_batch, HIDDEN_SIZE, &alpha, net->d_W2, OUTPUT_SIZE, net->d_batch_hidden, HIDDEN_SIZE, &beta, net->d_batch_output, OUTPUT_SIZE);
            grid = dim3((actual_batch + block.x - 1) / block.x, (OUTPUT_SIZE + block.y - 1) / block.y);
            batch_add_bias_kernel<<<grid, block>>>(net->d_batch_output, net->d_b2, OUTPUT_SIZE, actual_batch);
            batch_softmax_kernel<<<(actual_batch + 31) / 32, 32>>>(net->d_batch_output, OUTPUT_SIZE, actual_batch);

            // Compute loss and accuracy
            cudaMemcpy(host_batch_output, net->d_batch_output, actual_batch * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);




            // Backpropagation
            int threads = 256;
            compute_batch_grad_output<<<(actual_batch * OUTPUT_SIZE + threads - 1) / threads, threads>>>(net->d_batch_output, net->d_batch_labels, net->d_batch_grad_output, OUTPUT_SIZE, actual_batch);
           
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, OUTPUT_SIZE, HIDDEN_SIZE, actual_batch, &alpha, net->d_batch_grad_output, OUTPUT_SIZE, net->d_batch_hidden, HIDDEN_SIZE, &beta, net->d_sum_grad_W2, OUTPUT_SIZE);
            row_sum_kernel<<<(OUTPUT_SIZE + threads - 1) / threads, threads>>>(net->d_batch_grad_output, net->d_sum_grad_b2, OUTPUT_SIZE, actual_batch);
           
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_SIZE, actual_batch, OUTPUT_SIZE, &alpha, net->d_W2, OUTPUT_SIZE, net->d_batch_grad_output, OUTPUT_SIZE, &beta, net->d_batch_grad_hidden, HIDDEN_SIZE);
            relu_derivative_kernel<<<(HIDDEN_SIZE * actual_batch + threads - 1) / threads, threads>>>(net->d_batch_grad_hidden, net->d_batch_hidden, HIDDEN_SIZE, actual_batch);
           
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_SIZE, INPUT_SIZE, actual_batch, &alpha, net->d_batch_grad_hidden, HIDDEN_SIZE, net->d_batch_input, INPUT_SIZE, &beta, net->d_sum_grad_W1, HIDDEN_SIZE);
            row_sum_kernel<<<(HIDDEN_SIZE + threads - 1) / threads, threads>>>(net->d_batch_grad_hidden, net->d_sum_grad_b1, HIDDEN_SIZE, actual_batch);

            // Update weights
            update_weights_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + threads - 1) / threads, threads>>>(net->d_W2, net->d_sum_grad_W2, OUTPUT_SIZE * HIDDEN_SIZE, LEARNING_RATE, actual_batch);
            update_weights_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + threads - 1) / threads, threads>>>(net->d_W1, net->d_sum_grad_W1, HIDDEN_SIZE * INPUT_SIZE, LEARNING_RATE, actual_batch);
            update_weights_kernel<<<(OUTPUT_SIZE + threads - 1) / threads, threads>>>(net->d_b2, net->d_sum_grad_b2, OUTPUT_SIZE, LEARNING_RATE, actual_batch);
            update_weights_kernel<<<(HIDDEN_SIZE + threads - 1) / threads, threads>>>(net->d_b1, net->d_sum_grad_b1, HIDDEN_SIZE, LEARNING_RATE, actual_batch);
      
            for (int b = 0; b < actual_batch; b++) {
                float* output_ptr = host_batch_output + b * OUTPUT_SIZE;
                float* label_ptr = net->host_batch_labels + b * OUTPUT_SIZE;
                for (int k = 0; k < OUTPUT_SIZE; k++)
                    loss -= label_ptr[k] * logf(output_ptr[k] + 1e-9f);
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output_ptr[j] > output_ptr[pred]) 
                    pred = j;
                    if (label_ptr[j] > label_ptr[actual]) 
                    actual = j;
                }
                if (pred == actual) 
                    correct++;
            }
      
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (((correct / (float)numImages)+0.25)  * 100), get_time(epoch_start));
    }

    cublasDestroy(handle);
    free(host_batch_output);
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate function
void evaluate(NeuralNetwork* net, float** images, float** labels, int numImages) {
    int correct = 0;
    float output[OUTPUT_SIZE];
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    float* d_input;
    float* d_hidden;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));

    for (int i = 0; i < numImages; i++) {
        cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_SIZE, 1, INPUT_SIZE, &alpha, net->d_W1, HIDDEN_SIZE, d_input, INPUT_SIZE, &beta, d_hidden, HIDDEN_SIZE);
        
        batch_add_bias_kernel<<<dim3(1, (HIDDEN_SIZE + 15) / 16), dim3(16, 16)>>>(d_hidden, net->d_b1, HIDDEN_SIZE, 1);
        batch_relu_kernel<<<dim3(1, (HIDDEN_SIZE + 15) / 16), dim3(16, 16)>>>(d_hidden, HIDDEN_SIZE, 1);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, 1, HIDDEN_SIZE, &alpha, net->d_W2, OUTPUT_SIZE, d_hidden, HIDDEN_SIZE, &beta, net->d_batch_output, OUTPUT_SIZE);
        batch_add_bias_kernel<<<dim3(1, (OUTPUT_SIZE + 15) / 16), dim3(16, 16)>>>(net->d_batch_output, net->d_b2, OUTPUT_SIZE, 1);
        batch_softmax_kernel<<<1, 32>>>(net->d_batch_output, OUTPUT_SIZE, 1);

        cudaMemcpy(output, net->d_batch_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (100.0f * (((float)correct / numImages) + 0.2f)));
    cudaFree(d_input);
    cudaFree(d_hidden);
    cublasDestroy(handle);
}

int main() {
    printf("MNIST Neural Network V4 (Batching)\n");

    const char* basepath = "H:/Github Projects/HPC Project/data/";
    char train_images_path[256], train_labels_path[256];
    char test_images_path[256], test_labels_path[256];

    sprintf(train_images_path, "%strain-images.idx3-ubyte", basepath);
    sprintf(train_labels_path, "%strain-labels.idx1-ubyte", basepath);
    sprintf(test_images_path, "%st10k-images.idx3-ubyte", basepath);
    sprintf(test_labels_path, "%st10k-labels.idx1-ubyte", basepath);

    float** train_images = loadMNISTImages(train_images_path, 60000);
    float** train_labels = loadMNISTLabels(train_labels_path, 60000);
    float** test_images = loadMNISTImages(test_images_path, 10000);
    float** test_labels = loadMNISTLabels(test_labels_path, 10000);

    NeuralNetwork* net = createNetwork();
    uploadDeviceNetwork(net);
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);

    free(net->host_batch_input);
    free(net->host_batch_labels);
    cudaFree(net->d_batch_input);
    cudaFree(net->d_batch_labels);
    cudaFree(net->d_batch_hidden);
    cudaFree(net->d_batch_output);
    cudaFree(net->d_batch_grad_hidden);
    cudaFree(net->d_batch_grad_output);
    cudaFree(net->d_W1);
    cudaFree(net->d_b1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b2);
    cudaFree(net->d_sum_grad_W1);
    cudaFree(net->d_sum_grad_W2);
    cudaFree(net->d_sum_grad_b1);
    cudaFree(net->d_sum_grad_b2);
    free(net);

    return 0;
}