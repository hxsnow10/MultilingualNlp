#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_SYNONYM_NUM 10
#define LINE_MAX_NUM 100000
#define ADD_MAX 100000
char a[100][100];
int i;
char add_words[ADD_MAX][MAX_STRING];

int main(int argc, char* argv[]){
    long long n,m;
    float c;
    FILE *fin;
    char word[MAX_STRING];
    fin = fopen("text", "r");
    if (fin == NULL){
        printf("ERROR: freeze vec file not found!\n");
        exit(1);
    }
    fscanf(fin, "%lld", &n);
    fscanf(fin, "%lld", &m);
    fscanf(fin, "%f", &c);
    printf("%lld %lld %f",n,m, c);
}
