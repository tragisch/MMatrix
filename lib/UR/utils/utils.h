#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

/**
 * @brief write the content of a file in a string.
 *
 * @param path
 * @return char*
 */
char *file2string(char *path);

/**
 * @brief write string to file (from stackoverflow)
 *
 * @param filepath
 * @param data
 * @return int '0' if suceeded
 */
int string2file(char *filepath, char *data);

/**
 * @brief write a string of chars between start an end to string buffer
 *
 * @param str string
 * @param buffer result
 * @param start start char
 * @param end end char
 */
void slice(const char *str, char *buffer, size_t start, size_t end);

#endif