#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"


static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[MELVEC_LENGTH];

static uint32_t packet_cnt = 0;
uint8_t remain = 0;

static volatile int32_t rem_n_bufs = 0;

int StartADCAcq() {											//supression arg nombre de buffer => main
	cur_melvec = 0;
	if (1) {
		return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
	} else {
		return HAL_OK;
	}
}

int IsADCFinished(void) {
	return (0);
}
/*
static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
	//start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	DEBUG_PRINT("FV #:\t");
	for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
		DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[i]));
	}
	DEBUG_PRINT("\r\n");
	//stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}
*/
static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
	// BE encoding of each mel coef
	for (size_t j=0; j<MELVEC_LENGTH; j++) {
		(packet+PACKET_HEADER_LENGTH)[j*2]   = mel_vectors[j] >> 8;
		(packet+PACKET_HEADER_LENGTH)[j*2+1] = mel_vectors[j] & 0xFF;
	}
	// Write header and tag into the packet.
	make_packet(packet, PAYLOAD_LENGTH, 0, *packet_cnt);
	*packet_cnt += 1;
	if (*packet_cnt == 0) {
		// Should not happen as packet_cnt is 32-bit and we send at most 1 packet per second.
		DEBUG_PRINT("Packet counter overflow.\r\n");
		Error_Handler();
	}
}

static void send_spectrogram() {
	uint8_t packet[PACKET_LENGTH];
	q15_t bound;
	if (THRESHOLD_MOD) bound = 300; else bound = 70;
	if (vmax_global<bound && remain == 0){
		return;
	}
	if (remain ==0){
		remain = N_MELVECS-1;
	}else{
		remain --;
	}
	vmax_global=0;
	start_cycle_count();
	encode_packet(packet, &packet_cnt);
	stop_cycle_count("encode");
	S2LP_Send(packet, PACKET_LENGTH);
	//print_encoded_packet(packet);
}

static void ADC_Callback(int buf_cplt) {
	ADCDataRdy[buf_cplt] = 1;
	Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
	Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors);		//mel vector devient simple *20 delete
	ADCDataRdy[buf_cplt] = 0;
	send_spectrogram();
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	HAL_ResumeTick();
	ADC_Callback(1);

}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	HAL_ResumeTick();
	ADC_Callback(0);
}
