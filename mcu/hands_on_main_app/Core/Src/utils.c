/*
 * utils.c
 */
#include "config.h"
#include "stm32l4xx_hal.h"
#include "main.h"


// Encode the binary buffer buf of length len in the null-terminated string s
// (which must have length at least 2*len+1).
void hex_encode(char* s, const uint8_t* buf, size_t len) {
    s[2*len] = '\0';
    for (size_t i=0; i<len; i++) {
        s[i*2] = "0123456789abcdef"[buf[i] >> 4];
        s[i*2+1] = "0123456789abcdef"[buf[i] & 0xF];
    }
}
