/*
 * config.h
 *
 *  Created on: Oct 8, 2021
 *      Author: Teaching Assistants of LELEC210x
 */

#ifndef INC_CONFIG_H_
#define INC_CONFIG_H_


// Spectrogram parameters
#define SAMPLES_PER_MELVEC 512*20 // Dans les faits, c'est 512*20*2. Le *2 vient de l'initialisation ligne 60 de main.c
#define MELVEC_LENGTH 20
#define N_MELVECS 20
#define NBR_MESURES 260	//time = samples_per_melvec/fs*NBR_mesures (~2m pour l'instant)


#endif /* INC_CONFIG_H_ */
