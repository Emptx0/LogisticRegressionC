#include <stdio.h>

float train_data[][3] = {
        {10, 50, -1},
        {20, 30, 1},
        {25, 30, 1},
        {20, 60, -1}
};
#define train_count (sizeof(train_data) / sizeof(train_data[0]))

float test_data[][3] = {
        {40, 40, 1},
        {30, 45, 1},
        {20, 45, -1},
        {40, 30, 1},
        {7, 35, -1},
        {15, 70, -1}
};
#define test_count (sizeof(test_data) / sizeof(test_data[0]))

float activation_func(float z) {
    if (z > 0)
        return 1;
    else
        return -1;
}

float cost(float w1, float w2, float bias, float prediction[]) {
    float c = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float X1 = train_data[i][0];
        float X2 = train_data[i][1];
        float y = train_data[i][2];
        float y_pr = w1*X1 + w2*X2 + bias;

        prediction[i] = y_pr;

        float er = y - y_pr;
        c += er*er;
    }
    c /= train_count;
    return c;
}

int main() {
    float w[2] = {0.0f, 0.0f};
    float bias = 0.0f;
    float eps = 1e-5;
    float rate = 1e-4;
    int N = 1000;

    float prediction[6];

    for (size_t i = 0; i < N; ++i) {
        float c = cost(w[0], w[1], bias, prediction);
        float gradient_weight1 = (cost(w[0] + eps, w[1], bias, prediction) - c) / eps;
        float gradient_weight2 = (cost(w[0], w[1] + eps, bias, prediction) - c) / eps;
        float gradient_bias = (cost(w[0], w[1], bias + eps, prediction) - c) / eps;

        w[0] -= rate * gradient_weight1;
        w[1] -= rate * gradient_weight2;
        bias -= rate * gradient_bias;

        printf("w: { %f, %f, %f }; Cost = %f.\n", w[0], w[1], bias, c);
    }

    printf("\nPrediciton: \n");
    float mse = 0.0f;
    for (size_t i = 0; i < test_count; ++i) {
        prediction[i] = w[0] * test_data[i][0] + w[1] * test_data[i][1];
        printf("af: %f\n", activation_func(prediction[i]));
        float y = test_data[i][2];
        float error = (y - prediction[i])*(y - prediction[i]);
        mse += error;
    }
    mse /= test_count;
    printf("\nMSE: %f\n", mse);

    return 0;
}