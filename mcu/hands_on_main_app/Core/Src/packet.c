/*
 * packet.c
 */

#include "aes_ref.h"
#include "aes.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};


void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	// Allocate a buffer of the key size to store the input and result of AES
	// uint32_t[4] is 4*(32/8)= 16 bytes long

	uint32_t statew2[4] = {0};
	// state is a pointer to the start of the buffer
	uint8_t *state2 = (uint8_t*) statew2;
    size_t i;
    //size must be multiple of 16, complete with 0 msg
    /*uint8_t padding = 16-(msg_len%16);
    uint8_t input[msg_len+padding];
    for (size_t a =0; a<msg_len; a++){
    	input[a]=msg[a];
    	printf("%d\t",input[a]);
    }
    printf("\n");
    HAL_CRYP_AESCBC_Encrypt(&hcryp, input, msg_len+padding, tag, 1000);*/

    // TO DO : Complete the CBC-MAC_AES

	for (i = 0; i < msg_len-16; i +=16){
		for(size_t j = 0; j < 16; j++){
			*(state2+j) = msg[i+j] ^ *(state2+j);
		}
		HAL_CRYP_AESECB_Encrypt(&hcryp, state2, 16, state2, 1000);
	}
	for(size_t k = i; k < msg_len; k++){
		*(state2+k-i) = msg[k] ^ *(state2+k-i);
	}
	HAL_CRYP_AESECB_Encrypt(&hcryp, state2, 16, state2, 1000);

    // Copy the result of CBC-MAC-AES to the tag.
    for (int j=0; j<16; j++) {
        tag[j] = state2[j];
    }
    printf("end\n");
}
// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
    // Initially, the whole packet header is set to 0s

    memset(packet, 0, PACKET_HEADER_LENGTH);
    // So is the tag
	memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);
	packet[0] = 0;
	packet[1] = sender_id;
	packet[2] =(uint8_t) (payload_len>>8) & 0x00FF;
	packet[3] =(uint8_t) (payload_len)& 0x00FF;
	packet[4] =(uint8_t) (serial>>24)& 0x00FF;
	packet[5] =(uint8_t) (serial>>16)& 0x00FF;
	packet[6] =(uint8_t) (serial>>8)& 0x00FF;
	packet[7] =(uint8_t) (serial)& 0x00FF;


	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/

	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
