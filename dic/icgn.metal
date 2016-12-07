//
//  fftcc2d.metal
//  padic
//
//  Created by Tue Le on 9/25/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;

#define int_t int
#define real_t float
#define real_t2 float2
#define real_t4 float4

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_64 64

#define sync_threads() threadgroup_barrier(mem_flags::mem_threadgroup)

constant real_t c_dBicubicMatrix[16][16] = {
	{	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 },
	{	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 },
	{  -3,	3,	0,	0, -2, -1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 },
	{	2, -2,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 },
	{	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0 },
	{	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0 },
	{	0,	0,	0,	0,	0,	0,	0,	0, -3,	3,	0,	0, -2, -1,	0,	0 },
	{	0,	0,	0,	0,	0,	0,	0,	0,	2, -2,	0,	0,	1,	1,	0,	0 },
	{  -3,	0,	3,	0,	0,	0,	0,	0, -2,	0, -1,	0,	0,	0,	0,	0 },
	{	0,	0,	0,	0, -3,	0,	3,	0,	0,	0,	0,	0, -2,  0, -1,	0 },
	{	9, -9, -9,	9,	6,	3, -6, -3,	6, -6,	3, -3,	4,	2,	2,	1 },
	{  -6,	6,	6, -6, -3, -3,	3,	3, -4,	4, -2,	2, -2, -2, -1, -1 },
	{	2,	0, -2,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0 },
	{	0,	0,	0,	0,	2,	0, -2,	0,	0,	0,	0,	0,	1,	0,	1,	0 },
	{  -6,	6,	6, -6, -4, -2,	4,	2, -3,	3, -3,	3, -2, -1, -2, -1 },
	{	4, -4, -4,	4,	2,	2, -2, -2,	2, -2,	2, -2,	1,	1,	1,	1 }
};

template <int blockSize, class Real>
void reduceBlock(
  threadgroup Real *sdata,
  Real mySum,
  const unsigned int tid) {
  
//	sdata[tid] = mySum;
//  sync_threads();
//
//	// do reduction in shared mem
//	if (blockSize >= 512){ if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } sync_threads(); }
//	if (blockSize >= 256){ if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } sync_threads(); }
//	if (blockSize >= 128){ if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } sync_threads(); }
//
//	if (tid < 32) {
//		if (blockSize >= 64){ sdata[tid] = mySum = mySum + sdata[tid + 32]; }
//		if (blockSize >= 32){ sdata[tid] = mySum = mySum + sdata[tid + 16]; }
//		if (blockSize >= 16){ sdata[tid] = mySum = mySum + sdata[tid + 8]; }
//		if (blockSize >= 8)	{ sdata[tid] = mySum = mySum + sdata[tid + 4]; }
//		if (blockSize >= 4)	{ sdata[tid] = mySum = mySum + sdata[tid + 2]; }
//		if (blockSize >= 2) { sdata[tid] = mySum = mySum + sdata[tid + 1]; }
//	}
//
//  sync_threads();

	sdata[tid] = mySum;
  sync_threads();
  
  if (tid == 0) {
    for (int i = 1; i < blockSize; i++) {
      sdata[0] += sdata[i];
    }
  }
  sync_threads();
}

kernel void gradientXY2ImagesMetal(
  // Inputs.
  const device uchar *fImgF [[ buffer(0) ]],
  const device uchar *fImgG [[ buffer(1) ]],
  const device int_t &iStartX_in [[ buffer(2) ]],
  const device int_t &iStartY_in [[ buffer(3) ]],
  const device int_t &iROIWidth_in [[ buffer(4) ]],
  const device int_t &iROIHeight_in [[ buffer(5) ]],
  const device int_t &iImgWidth_in [[ buffer(6) ]],
  const device int_t &iImgHeight_in [[buffer(7) ]],
  // Outputs.
  device real_t *Fx [[ buffer(8) ]],
  device real_t *Fy [[ buffer(9) ]],
  device real_t *Gx [[ buffer(10) ]],
  device real_t *Gy [[ buffer(11) ]],
  device real_t *Gxy [[ buffer(12) ]],
  // Thread params.
  uint2 threadIdx [[ thread_position_in_threadgroup ]],
  uint2 blockDim [[ threads_per_threadgroup ]],
  uint2 blockIdx [[ threadgroup_position_in_grid ]]) {
  
  const int iStartX = iStartX_in;
  const int iStartY = iStartY_in;
  const int iROIWidth = iROIWidth_in;
  const int iROIHeight = iROIHeight_in;
  const int iImgWidth = iImgWidth_in;
  const int iImgHeight = iImgHeight_in;
  
	// Block Index
	const int by = blockIdx.y;
	const int bx = blockIdx.x;

	// Thread Index
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;

	// Global Memory offset: every block actually begin with 2 overlapped pixels
	const int y = iStartY - 1 + ty + (BLOCK_SIZE_Y - 2) * by;
	const int x = iStartX - 1 + tx + (BLOCK_SIZE_X - 2) * bx;

	// Declare the shared memory for storing the tiled subset
	threadgroup real_t imgF_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	threadgroup real_t imgG_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	// 1D index of the outpus
	int ind = 0;

	// Load the images into shared memory
	if (y < iStartY + iROIHeight + 1 && x < iStartX + iROIWidth + 1) {
		imgF_sh[ty][tx] = (real_t)fImgF[y * iImgWidth + x];
		imgG_sh[ty][tx] = (real_t)fImgG[y * iImgWidth + x];
	}
  
  sync_threads();

	// Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
	if (y >= iStartY && y < iROIHeight + iStartY && x >= iStartX && x < iROIWidth + iStartX &&
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1) {
    
		ind = (y - iStartY)*iROIWidth + (x - iStartX);
		Fx[ind] = 0.5 * (imgF_sh[ty][tx + 1] - imgF_sh[ty][tx - 1]);
		Fy[ind] = 0.5 * (imgF_sh[ty + 1][tx] - imgF_sh[ty - 1][tx]);

		Gx[ind] = 0.5 * (imgG_sh[ty][tx + 1] - imgG_sh[ty][tx - 1]);
		Gy[ind] = 0.5 * (imgG_sh[ty + 1][tx] - imgG_sh[ty - 1][tx]);
		Gxy[ind]= 0.25* (imgG_sh[ty + 1][tx + 1] - imgG_sh[ty - 1][tx + 1] - imgG_sh[ty + 1][tx - 1] + imgG_sh[ty - 1][tx - 1]);
	}
}

kernel void bicubicCoefficientsMetal(
  // Inputs.
  const device uchar* dIn_fImgT [[ buffer(0) ]],
  const device real_t* dIn_fTx [[ buffer(1) ]],
  const device real_t* dIn_fTy [[ buffer(2) ]],
  const device real_t* dIn_fTxy [[ buffer(3) ]],
  const device int &iStartX_in [[ buffer(4) ]],
  const device int &iStartY_in [[ buffer(5) ]],
  const device int &iROIWidth_in [[ buffer(6) ]],
  const device int &iROIHeight_in [[ buffer(7) ]],
  const device int &iImgWidth_in [[ buffer(8) ]],
  const device int &iImgHeight_in [[ buffer(9) ]],
  // Outputs.
  device real_t4* dOut_fBicubicInterpolants [[ buffer(10) ]],
  // Thread params.
  uint2 threadIdx [[ thread_position_in_threadgroup ]],
  uint2 blockDim [[ threads_per_threadgroup ]],
  uint2 blockIdx [[ threadgroup_position_in_grid ]]) {
  
  int iStartX = iStartX_in;
  int iStartY = iStartY_in;
  int iROIWidth = iROIWidth_in;
  int iROIHeight = iROIHeight_in;
  int iImgWidth = iImgWidth_in;
  int iImgHeight = iImgHeight_in;
  
 
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

	// These two temporary arrays may consult 32 registers,
	// half of the allowed ones for each thread's
	float fAlphaT[16], fTaoT[16];

	if (y < iROIHeight - 1 && x < iROIWidth - 1) {
  
		fTaoT[0] = (real_t)dIn_fImgT[(y + iStartY)*iImgWidth + iStartX + x];
		fTaoT[1] = (real_t)dIn_fImgT[(y + iStartY)*iImgWidth + iStartX + x + 1];
		fTaoT[2] = (real_t)dIn_fImgT[(y + 1 + iStartY)*iImgWidth + iStartX + x];
		fTaoT[3] = (real_t)dIn_fImgT[(y + 1 + iStartY)*iImgWidth + iStartX + x + 1];
		fTaoT[4] = dIn_fTx[y*iROIWidth + x];
		fTaoT[5] = dIn_fTx[y*iROIWidth + x + 1];
		fTaoT[6] = dIn_fTx[(y + 1)*iROIWidth + x];
		fTaoT[7] = dIn_fTx[(y + 1)*iROIWidth + x + 1];
		fTaoT[8] = dIn_fTy[y*iROIWidth + x];
		fTaoT[9] = dIn_fTy[y*iROIWidth + x + 1];
		fTaoT[10] = dIn_fTy[(y + 1)*iROIWidth + x];
		fTaoT[11] = dIn_fTy[(y + 1)*iROIWidth + x + 1];
		fTaoT[12] = dIn_fTxy[y*iROIWidth + x];
		fTaoT[13] = dIn_fTxy[y*iROIWidth + x + 1];
		fTaoT[14] = dIn_fTxy[(y + 1)*iROIWidth + x];
		fTaoT[15] = dIn_fTxy[(y + 1)*iROIWidth + x + 1];

		// Reduction to calculate fAlphaT (unroll the "for" loop).
		fAlphaT[0] = c_dBicubicMatrix[0][0] * fTaoT[0] + c_dBicubicMatrix[0][1] * fTaoT[1] + c_dBicubicMatrix[0][2] * fTaoT[2] + c_dBicubicMatrix[0][3] * fTaoT[3] +
			c_dBicubicMatrix[0][4] * fTaoT[4] + c_dBicubicMatrix[0][5] * fTaoT[5] + c_dBicubicMatrix[0][6] * fTaoT[6] + c_dBicubicMatrix[0][7] * fTaoT[7] +
			c_dBicubicMatrix[0][8] * fTaoT[8] + c_dBicubicMatrix[0][9] * fTaoT[9] + c_dBicubicMatrix[0][10] * fTaoT[10] + c_dBicubicMatrix[0][11] * fTaoT[11] +
			c_dBicubicMatrix[0][12] * fTaoT[12] + c_dBicubicMatrix[0][13] * fTaoT[13] + c_dBicubicMatrix[0][14] * fTaoT[14] + c_dBicubicMatrix[0][15] * fTaoT[15];
		fAlphaT[1] = c_dBicubicMatrix[1][0] * fTaoT[0] + c_dBicubicMatrix[1][1] * fTaoT[1] + c_dBicubicMatrix[1][2] * fTaoT[2] + c_dBicubicMatrix[1][3] * fTaoT[3] +
			c_dBicubicMatrix[1][4] * fTaoT[4] + c_dBicubicMatrix[1][5] * fTaoT[5] + c_dBicubicMatrix[1][6] * fTaoT[6] + c_dBicubicMatrix[1][7] * fTaoT[7] +
			c_dBicubicMatrix[1][8] * fTaoT[8] + c_dBicubicMatrix[1][9] * fTaoT[9] + c_dBicubicMatrix[1][10] * fTaoT[10] + c_dBicubicMatrix[1][11] * fTaoT[11] +
			c_dBicubicMatrix[1][12] * fTaoT[12] + c_dBicubicMatrix[1][13] * fTaoT[13] + c_dBicubicMatrix[1][14] * fTaoT[14] + c_dBicubicMatrix[1][15] * fTaoT[15];
		fAlphaT[2] = c_dBicubicMatrix[2][0] * fTaoT[0] + c_dBicubicMatrix[2][1] * fTaoT[1] + c_dBicubicMatrix[2][2] * fTaoT[2] + c_dBicubicMatrix[2][3] * fTaoT[3] +
			c_dBicubicMatrix[2][4] * fTaoT[4] + c_dBicubicMatrix[2][5] * fTaoT[5] + c_dBicubicMatrix[2][6] * fTaoT[6] + c_dBicubicMatrix[2][7] * fTaoT[7] +
			c_dBicubicMatrix[2][8] * fTaoT[8] + c_dBicubicMatrix[2][9] * fTaoT[9] + c_dBicubicMatrix[2][10] * fTaoT[10] + c_dBicubicMatrix[2][11] * fTaoT[11] +
			c_dBicubicMatrix[2][12] * fTaoT[12] + c_dBicubicMatrix[2][13] * fTaoT[13] + c_dBicubicMatrix[2][14] * fTaoT[14] + c_dBicubicMatrix[2][15] * fTaoT[15];
		fAlphaT[3] = c_dBicubicMatrix[3][0] * fTaoT[0] + c_dBicubicMatrix[3][1] * fTaoT[1] + c_dBicubicMatrix[3][2] * fTaoT[2] + c_dBicubicMatrix[3][3] * fTaoT[3] +
			c_dBicubicMatrix[3][4] * fTaoT[4] + c_dBicubicMatrix[3][5] * fTaoT[5] + c_dBicubicMatrix[3][6] * fTaoT[6] + c_dBicubicMatrix[3][7] * fTaoT[7] +
			c_dBicubicMatrix[3][8] * fTaoT[8] + c_dBicubicMatrix[3][9] * fTaoT[9] + c_dBicubicMatrix[3][10] * fTaoT[10] + c_dBicubicMatrix[3][11] * fTaoT[11] +
			c_dBicubicMatrix[3][12] * fTaoT[12] + c_dBicubicMatrix[3][13] * fTaoT[13] + c_dBicubicMatrix[3][14] * fTaoT[14] + c_dBicubicMatrix[3][15] * fTaoT[15];
		fAlphaT[4] = c_dBicubicMatrix[4][0] * fTaoT[0] + c_dBicubicMatrix[4][1] * fTaoT[1] + c_dBicubicMatrix[4][2] * fTaoT[2] + c_dBicubicMatrix[4][3] * fTaoT[3] +
			c_dBicubicMatrix[4][4] * fTaoT[4] + c_dBicubicMatrix[4][5] * fTaoT[5] + c_dBicubicMatrix[4][6] * fTaoT[6] + c_dBicubicMatrix[4][7] * fTaoT[7] +
			c_dBicubicMatrix[4][8] * fTaoT[8] + c_dBicubicMatrix[4][9] * fTaoT[9] + c_dBicubicMatrix[4][10] * fTaoT[10] + c_dBicubicMatrix[4][11] * fTaoT[11] +
			c_dBicubicMatrix[4][12] * fTaoT[12] + c_dBicubicMatrix[4][13] * fTaoT[13] + c_dBicubicMatrix[4][14] * fTaoT[14] + c_dBicubicMatrix[4][15] * fTaoT[15];
		fAlphaT[5] = c_dBicubicMatrix[5][0] * fTaoT[0] + c_dBicubicMatrix[5][1] * fTaoT[1] + c_dBicubicMatrix[5][2] * fTaoT[2] + c_dBicubicMatrix[5][3] * fTaoT[3] +
			c_dBicubicMatrix[5][4] * fTaoT[4] + c_dBicubicMatrix[5][5] * fTaoT[5] + c_dBicubicMatrix[5][6] * fTaoT[6] + c_dBicubicMatrix[5][7] * fTaoT[7] +
			c_dBicubicMatrix[5][8] * fTaoT[8] + c_dBicubicMatrix[5][9] * fTaoT[9] + c_dBicubicMatrix[5][10] * fTaoT[10] + c_dBicubicMatrix[5][11] * fTaoT[11] +
			c_dBicubicMatrix[5][12] * fTaoT[12] + c_dBicubicMatrix[5][13] * fTaoT[13] + c_dBicubicMatrix[5][14] * fTaoT[14] + c_dBicubicMatrix[5][15] * fTaoT[15];
		fAlphaT[6] = c_dBicubicMatrix[6][0] * fTaoT[0] + c_dBicubicMatrix[6][1] * fTaoT[1] + c_dBicubicMatrix[6][2] * fTaoT[2] + c_dBicubicMatrix[6][3] * fTaoT[3] +
			c_dBicubicMatrix[6][4] * fTaoT[4] + c_dBicubicMatrix[6][5] * fTaoT[5] + c_dBicubicMatrix[6][6] * fTaoT[6] + c_dBicubicMatrix[6][7] * fTaoT[7] +
			c_dBicubicMatrix[6][8] * fTaoT[8] + c_dBicubicMatrix[6][9] * fTaoT[9] + c_dBicubicMatrix[6][10] * fTaoT[10] + c_dBicubicMatrix[6][11] * fTaoT[11] +
			c_dBicubicMatrix[6][12] * fTaoT[12] + c_dBicubicMatrix[6][13] * fTaoT[13] + c_dBicubicMatrix[6][14] * fTaoT[14] + c_dBicubicMatrix[6][15] * fTaoT[15];
		fAlphaT[7] = c_dBicubicMatrix[7][0] * fTaoT[0] + c_dBicubicMatrix[7][1] * fTaoT[1] + c_dBicubicMatrix[7][2] * fTaoT[2] + c_dBicubicMatrix[7][3] * fTaoT[3] +
			c_dBicubicMatrix[7][4] * fTaoT[4] + c_dBicubicMatrix[7][5] * fTaoT[5] + c_dBicubicMatrix[7][6] * fTaoT[6] + c_dBicubicMatrix[7][7] * fTaoT[7] +
			c_dBicubicMatrix[7][8] * fTaoT[8] + c_dBicubicMatrix[7][9] * fTaoT[9] + c_dBicubicMatrix[7][10] * fTaoT[10] + c_dBicubicMatrix[7][11] * fTaoT[11] +
			c_dBicubicMatrix[7][12] * fTaoT[12] + c_dBicubicMatrix[7][13] * fTaoT[13] + c_dBicubicMatrix[7][14] * fTaoT[14] + c_dBicubicMatrix[7][15] * fTaoT[15];
		fAlphaT[8] = c_dBicubicMatrix[8][0] * fTaoT[0] + c_dBicubicMatrix[8][1] * fTaoT[1] + c_dBicubicMatrix[8][2] * fTaoT[2] + c_dBicubicMatrix[8][3] * fTaoT[3] +
			c_dBicubicMatrix[8][4] * fTaoT[4] + c_dBicubicMatrix[8][5] * fTaoT[5] + c_dBicubicMatrix[8][6] * fTaoT[6] + c_dBicubicMatrix[8][7] * fTaoT[7] +
			c_dBicubicMatrix[8][8] * fTaoT[8] + c_dBicubicMatrix[8][9] * fTaoT[9] + c_dBicubicMatrix[8][10] * fTaoT[10] + c_dBicubicMatrix[8][11] * fTaoT[11] +
			c_dBicubicMatrix[8][12] * fTaoT[12] + c_dBicubicMatrix[8][13] * fTaoT[13] + c_dBicubicMatrix[8][14] * fTaoT[14] + c_dBicubicMatrix[8][15] * fTaoT[15];
		fAlphaT[9] = c_dBicubicMatrix[9][0] * fTaoT[0] + c_dBicubicMatrix[9][1] * fTaoT[1] + c_dBicubicMatrix[9][2] * fTaoT[2] + c_dBicubicMatrix[9][3] * fTaoT[3] +
			c_dBicubicMatrix[9][4] * fTaoT[4] + c_dBicubicMatrix[9][5] * fTaoT[5] + c_dBicubicMatrix[9][6] * fTaoT[6] + c_dBicubicMatrix[9][7] * fTaoT[7] +
			c_dBicubicMatrix[9][8] * fTaoT[8] + c_dBicubicMatrix[9][9] * fTaoT[9] + c_dBicubicMatrix[9][10] * fTaoT[10] + c_dBicubicMatrix[9][11] * fTaoT[11] +
			c_dBicubicMatrix[9][12] * fTaoT[12] + c_dBicubicMatrix[9][13] * fTaoT[13] + c_dBicubicMatrix[9][14] * fTaoT[14] + c_dBicubicMatrix[9][15] * fTaoT[15];
		fAlphaT[10] = c_dBicubicMatrix[10][0] * fTaoT[0] + c_dBicubicMatrix[10][1] * fTaoT[1] + c_dBicubicMatrix[10][2] * fTaoT[2] + c_dBicubicMatrix[10][3] * fTaoT[3] +
			c_dBicubicMatrix[10][4] * fTaoT[4] + c_dBicubicMatrix[10][5] * fTaoT[5] + c_dBicubicMatrix[10][6] * fTaoT[6] + c_dBicubicMatrix[10][7] * fTaoT[7] +
			c_dBicubicMatrix[10][8] * fTaoT[8] + c_dBicubicMatrix[10][9] * fTaoT[9] + c_dBicubicMatrix[10][10] * fTaoT[10] + c_dBicubicMatrix[10][11] * fTaoT[11] +
			c_dBicubicMatrix[10][12] * fTaoT[12] + c_dBicubicMatrix[10][13] * fTaoT[13] + c_dBicubicMatrix[10][14] * fTaoT[14] + c_dBicubicMatrix[10][15] * fTaoT[15];
		fAlphaT[11] = c_dBicubicMatrix[11][0] * fTaoT[0] + c_dBicubicMatrix[11][1] * fTaoT[1] + c_dBicubicMatrix[11][2] * fTaoT[2] + c_dBicubicMatrix[11][3] * fTaoT[3] +
			c_dBicubicMatrix[11][4] * fTaoT[4] + c_dBicubicMatrix[11][5] * fTaoT[5] + c_dBicubicMatrix[11][6] * fTaoT[6] + c_dBicubicMatrix[11][7] * fTaoT[7] +
			c_dBicubicMatrix[11][8] * fTaoT[8] + c_dBicubicMatrix[11][9] * fTaoT[9] + c_dBicubicMatrix[11][10] * fTaoT[10] + c_dBicubicMatrix[11][11] * fTaoT[11] +
			c_dBicubicMatrix[11][12] * fTaoT[12] + c_dBicubicMatrix[11][13] * fTaoT[13] + c_dBicubicMatrix[11][14] * fTaoT[14] + c_dBicubicMatrix[11][15] * fTaoT[15];
		fAlphaT[12] = c_dBicubicMatrix[12][0] * fTaoT[0] + c_dBicubicMatrix[12][1] * fTaoT[1] + c_dBicubicMatrix[12][2] * fTaoT[2] + c_dBicubicMatrix[12][3] * fTaoT[3] +
			c_dBicubicMatrix[12][4] * fTaoT[4] + c_dBicubicMatrix[12][5] * fTaoT[5] + c_dBicubicMatrix[12][6] * fTaoT[6] + c_dBicubicMatrix[12][7] * fTaoT[7] +
			c_dBicubicMatrix[12][8] * fTaoT[8] + c_dBicubicMatrix[12][9] * fTaoT[9] + c_dBicubicMatrix[12][10] * fTaoT[10] + c_dBicubicMatrix[12][11] * fTaoT[11] +
			c_dBicubicMatrix[12][12] * fTaoT[12] + c_dBicubicMatrix[12][13] * fTaoT[13] + c_dBicubicMatrix[12][14] * fTaoT[14] + c_dBicubicMatrix[12][15] * fTaoT[15];
		fAlphaT[13] = c_dBicubicMatrix[13][0] * fTaoT[0] + c_dBicubicMatrix[13][1] * fTaoT[1] + c_dBicubicMatrix[13][2] * fTaoT[2] + c_dBicubicMatrix[13][3] * fTaoT[3] +
			c_dBicubicMatrix[13][4] * fTaoT[4] + c_dBicubicMatrix[13][5] * fTaoT[5] + c_dBicubicMatrix[13][6] * fTaoT[6] + c_dBicubicMatrix[13][7] * fTaoT[7] +
			c_dBicubicMatrix[13][8] * fTaoT[8] + c_dBicubicMatrix[13][9] * fTaoT[9] + c_dBicubicMatrix[13][10] * fTaoT[10] + c_dBicubicMatrix[13][11] * fTaoT[11] +
			c_dBicubicMatrix[13][12] * fTaoT[12] + c_dBicubicMatrix[13][13] * fTaoT[13] + c_dBicubicMatrix[13][14] * fTaoT[14] + c_dBicubicMatrix[13][15] * fTaoT[15];
		fAlphaT[14] = c_dBicubicMatrix[14][0] * fTaoT[0] + c_dBicubicMatrix[14][1] * fTaoT[1] + c_dBicubicMatrix[14][2] * fTaoT[2] + c_dBicubicMatrix[14][3] * fTaoT[3] +
			c_dBicubicMatrix[14][4] * fTaoT[4] + c_dBicubicMatrix[14][5] * fTaoT[5] + c_dBicubicMatrix[14][6] * fTaoT[6] + c_dBicubicMatrix[14][7] * fTaoT[7] +
			c_dBicubicMatrix[14][8] * fTaoT[8] + c_dBicubicMatrix[14][9] * fTaoT[9] + c_dBicubicMatrix[14][10] * fTaoT[10] + c_dBicubicMatrix[14][11] * fTaoT[11] +
			c_dBicubicMatrix[14][12] * fTaoT[12] + c_dBicubicMatrix[14][13] * fTaoT[13] + c_dBicubicMatrix[14][14] * fTaoT[14] + c_dBicubicMatrix[14][15] * fTaoT[15];
		fAlphaT[15] = c_dBicubicMatrix[15][0] * fTaoT[0] + c_dBicubicMatrix[15][1] * fTaoT[1] + c_dBicubicMatrix[15][2] * fTaoT[2] + c_dBicubicMatrix[15][3] * fTaoT[3] +
			c_dBicubicMatrix[15][4] * fTaoT[4] + c_dBicubicMatrix[15][5] * fTaoT[5] + c_dBicubicMatrix[15][6] * fTaoT[6] + c_dBicubicMatrix[15][7] * fTaoT[7] +
			c_dBicubicMatrix[15][8] * fTaoT[8] + c_dBicubicMatrix[15][9] * fTaoT[9] + c_dBicubicMatrix[15][10] * fTaoT[10] + c_dBicubicMatrix[15][11] * fTaoT[11] +
			c_dBicubicMatrix[15][12] * fTaoT[12] + c_dBicubicMatrix[15][13] * fTaoT[13] + c_dBicubicMatrix[15][14] * fTaoT[14] + c_dBicubicMatrix[15][15] * fTaoT[15];

		// Write the results back to the fBicubicInterpolants array.
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[0];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[1];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[2];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[3];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[4];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[5];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[6];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[7];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[8];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[9];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[10];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[11];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[12];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[13];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[14];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[15];
	}
}

kernel void refAllSubetsNormMetal(
  // Inputs.
  const device uchar* d_refImg [[ buffer(0) ]],
  const device int *d_iPOIXY [[ buffer(1) ]],
  const device int &iSubsetW_in [[ buffer(2) ]],
  const device int &iSubsetH_in [[ buffer(3) ]],
  const device int &iSubsetX_in [[ buffer(4) ]],
  const device int &iSubsetY_in [[ buffer(5) ]],
  const device int &iImgWidth_in [[ buffer(6) ]],
  const device int &iImgHeight_in [[ buffer(7) ]],
  // Outputs.
  device real_t *whole_dSubSet [[ buffer(8) ]],
  device real_t *whole_dSubsetAve [[ buffer(9) ]],
  // Thread params.
  uint2 threadIdx [[ thread_position_in_threadgroup ]],
  uint2 blockDim [[ threads_per_threadgroup ]],
  uint2 blockIdx [[ threadgroup_position_in_grid ]]) {
  
  int iSubsetW = iSubsetW_in;
  int iSubsetH = iSubsetH_in;
  int iSubsetX = iSubsetX_in;
  int iSubsetY = iSubsetY_in;
  int iImgWidth = iImgWidth_in;
  int iImgHeight = iImgHeight_in;
  
	threadgroup real_t sm[BLOCK_SIZE_64];
  
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int size = iSubsetH * iSubsetW;
  
	real_t avg;
	real_t mySum = 0;
	real_t tempt;
	device real_t *fSubSet = whole_dSubSet + size * bid;
	device real_t *fSubsetAve = whole_dSubsetAve + (size + 1) * bid;
  
	for (int id = tid; id < size; id += dim) {
		int	l = id / iSubsetW;
		int m = id % iSubsetW;
		tempt = (real_t)d_refImg[int(d_iPOIXY[bid * 2] - iSubsetY + l) * iImgWidth + int(d_iPOIXY[bid * 2 + 1] - iSubsetX + m)];
		fSubSet[id] = tempt;
		mySum += tempt / size;
	}
  
  sync_threads();
  
	reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
  
  sync_threads();
  
	avg = sm[0];
	mySum = 0;
	for (int id = tid; id < size; id += dim) {
		tempt = fSubSet[id] - avg;
		mySum += tempt * tempt;
		fSubsetAve[id + 1] = tempt;
	}
  
  sync_threads();
  
	reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
  
  sync_threads();
  
	if (tid == 0) {
		fSubsetAve[0] = sqrt(sm[tid]);
	}
}

kernel void inverseHessianMetal(
  // Inputs.
  const device real_t *d_Rx [[ buffer(0) ]],
  const device real_t *d_Ry [[ buffer(1) ]],
  const device int *d_iPOIXY [[ buffer(2) ]],
  const device int &iSubsetX_in [[ buffer(3) ]],
  const device int &iSubsetY_in [[ buffer(4) ]],
  const device int &iSubsetW_in [[ buffer(5) ]],
  const device int &iSubsetH_in [[ buffer(6) ]],
  const device int &iStartX_in [[ buffer(7) ]],
  const device int &iStartY_in [[ buffer(8) ]],
  const device int &iROIWidth_in [[ buffer(9) ]],
  const device int &iROIHeight_in [[ buffer(10) ]],
  // Outputs.
  device real_t2 *whole_d_RDescent [[ buffer(11) ]],
  device real_t *whole_d_InvHessian [[ buffer(12) ]],
  device real_t *hes [[ buffer(13) ]],
  // Thread params.
  uint threadIdx [[ thread_position_in_threadgroup ]],
  uint blockDim [[ threads_per_threadgroup ]],
  uint blockIdx [[ threadgroup_position_in_grid ]]) {
  
  int iSubsetX = iSubsetX_in;
  int iSubsetY = iSubsetY_in;
  int iSubsetW = iSubsetW_in;
  int iSubsetH = iSubsetH_in;
  int iStartX = iStartX_in;
  int iStartY = iStartY_in;
  int iROIWidth = iROIWidth_in;
  int iROIHeight = iROIHeight_in;
 
	threadgroup real_t Hessian[96];
	threadgroup real_t sm[BLOCK_SIZE_64];
	threadgroup int iIndOfRowTempt[8];
  
	int tid = threadIdx;
	int dim = blockDim;
	int bid = blockIdx;
	int iSubWindowSize = iSubsetH * iSubsetW;
	int l;
	int m;

	real_t tempt;
	real_t t_dD0 = 0;
	real_t t_dD1 = 0;
	real_t t_dD2 = 0;
	real_t t_dD3 = 0;
	real_t t_dD4 = 0;
	real_t t_dD5 = 0;
  
	device real_t2* RDescent = whole_d_RDescent + bid*iSubWindowSize * 3;
	device real_t* r_dInvHessian = whole_d_InvHessian + bid * 36;
  
	for (int id = tid; id < 96; id += dim) {
		Hessian[id] = 0;
	}
  
  int maxIters = (iSubWindowSize + dim - 1) / dim;
  int id = tid;

  for (int iter = 0; iter < maxIters; iter++) {
    if (id < iSubWindowSize) {
  		l = id / iSubsetW;
  		m = id % iSubsetW;
      
  		real_t tx = d_Rx[iROIWidth * (d_iPOIXY[bid * 2] - iSubsetY + l - iStartY) + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
      
  		RDescent[l * iSubsetW + m].x = t_dD0 = tx;
  		RDescent[l * iSubsetW + m].y = t_dD1 = tx * (m - iSubsetX);
  		RDescent[iSubWindowSize + l * iSubsetW + m].x = t_dD2 = tx * (l - iSubsetY);

  		real_t ty = d_Ry[iROIWidth * (d_iPOIXY[bid * 2] - iSubsetY + l - iStartY) + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
      
  		RDescent[iSubWindowSize + l * iSubsetW + m].y = t_dD3 = ty;
  		RDescent[iSubWindowSize * 2 + l * iSubsetW + m].x = t_dD4 = ty * (m - iSubsetX);
  		RDescent[iSubWindowSize * 2 + l * iSubsetW + m].y = t_dD5 = ty * (l - iSubsetY);
    
      id += dim;
    } else {
      t_dD0 = t_dD1 = t_dD2 = t_dD3 = t_dD4 = t_dD5 = 0;
    }
    
		//00
		tempt = t_dD0 * t_dD0;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 0] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//11
		tempt = t_dD1 * t_dD1;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[1 * 16 + 1] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
    
		//22
		tempt = t_dD2 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[2 * 16 + 2] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//33
		tempt = t_dD3 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[3 * 16 + 3] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//44
		tempt = t_dD4 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[4 * 16 + 4] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//55
		tempt = t_dD5 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[5 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }


		//01
		tempt = t_dD0 * t_dD1;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 1] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//02
		tempt = t_dD0 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 2] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//03
		tempt = t_dD0 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 3] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//04
		tempt = t_dD0 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 4] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//05
		tempt = t_dD0 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[0 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }




		//12
		tempt = t_dD1 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[1 * 16 + 2] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//13
		tempt = t_dD1 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[1 * 16 + 3] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//14
		tempt = t_dD1 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[1 * 16 + 4] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//15
		tempt = t_dD1 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[1 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }



		//23
		tempt = t_dD2 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[2 * 16 + 3] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//24
		tempt = t_dD2 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[2 * 16 + 4] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//25
		tempt = t_dD2 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[2 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }


		//34
		tempt = t_dD3 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[3 * 16 + 4] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
		//35
		tempt = t_dD3 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[3 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }

		//45
		tempt = t_dD4 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0) {
			Hessian[4 * 16 + 5] += sm[0];
		} else {
      Hessian[95] += sm[0];
    }
    
    sync_threads();
		if (tid < BLOCK_SIZE_64) {
			sm[tid] = 0;
    }
	}
 
	if (tid < 5) {
		Hessian[(tid + 1) * 16 + 0] = Hessian[0 * 16 + (tid + 1)];
	}
  
	if (tid < 4) {
		Hessian[(tid + 2) * 16 + 1] = Hessian[1 * 16 + (tid + 2)];
	}
  
	if (tid < 3) {
		Hessian[(tid + 3) * 16 + 2] = Hessian[2 * 16 + (tid + 3)];
	}
  
	if (tid < 2) {
		Hessian[(tid + 4) * 16 + 3] = Hessian[3 * 16 + (tid + 4)];
	}
  
	if (tid == 0) {
		Hessian[5 * 16 + 4] = Hessian[4 * 16 + 5];
	}

	if (tid < 6) {
		Hessian[tid * 16 + tid + 8] = 1;
	}
  
  sync_threads();
  
	for (int l = 0; l < 6; l++) {
		// Find pivot (maximum lth column element) in the rest (6-l) rows
		if (tid < 8) {
			iIndOfRowTempt[tid] = l;
		}
    
    sync_threads();
    
		if (tid < 6 - l) {
			iIndOfRowTempt[tid] = tid + l;
		}
    
    sync_threads();

		if (tid < 4) {
			if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 4] * 16 + l]) {
				iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 4];
      }
		}
    
    sync_threads();
    
		if (tid < 2) {
			if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 2] * 16 + l]) {
				iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 2];
      }
		}
    
    sync_threads();
    
		if (tid == 0) {
			if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 1] * 16 + l]) {
				iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 1];
      }
      
			if (Hessian[iIndOfRowTempt[tid] * 16 + l] == 0) {
				Hessian[iIndOfRowTempt[tid] * 16 + l] = 0.0000001;
			}
		}
    
    sync_threads();
    
		if (tid < 12) {
			int m_iIndexOfCol = tid / 6 * 8 + tid % 6;
			real_t m_dTempt;
      
			if (iIndOfRowTempt[0] != l) {
      
				m_dTempt = Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol];
				Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol] = Hessian[l * 16 + m_iIndexOfCol];
				Hessian[l * 16 + m_iIndexOfCol] = m_dTempt;
			}
    }
    
    sync_threads();
    
    real_t div = Hessian[l * 16 + l];
    
    sync_threads();

		if (tid < 12) {
			int m_iIndexOfCol = tid / 6 * 8 + tid % 6;

			// Perform row operation to form required identity matrix out of the Hessian matrix

			Hessian[l * 16 + m_iIndexOfCol] /= div;
      
			for (int next_row = 0; next_row < 6; next_row++) {
				if (next_row != l) {
					Hessian[next_row * 16 + m_iIndexOfCol] -= Hessian[l * 16 + m_iIndexOfCol] * Hessian[next_row * 16 + l];
				}
			}
		}
    
    sync_threads();
	}

	// inv Hessian
	if (tid < 32) {
		r_dInvHessian[tid] = Hessian[tid / 6 * 16 + tid % 6 + 8];
  }
  
	if (tid < 4) {
		r_dInvHessian[tid + 32] = Hessian[(tid + 32) / 6 * 16 + (tid + 32) % 6 + 8];
	}
}

#define sqr(x) ((x) * (x))

kernel void icgnComputeMetal(
  // Inputs and outputs.
  device real_t *d_fU [[ buffer(0) ]],
  device real_t *d_fV [[ buffer(1) ]],
  // Inputs.
  const device int* d_iPOIXY [[ buffer(2) ]],
  const device int &iImgWidth_in [[ buffer(3) ]],
  const device int &iImgHeight_in [[ buffer(4) ]],
  const device int &iStartX_in [[ buffer(5) ]],
  const device int &iStartY_in [[ buffer(6) ]],
  const device int &iROIWidth_in [[ buffer(7) ]],
  const device int &iROIHeight_in [[ buffer(8) ]],
  const device int &iSubsetX_in [[ buffer(9) ]],
  const device int &iSubsetY_in [[ buffer(10) ]],
  const device int &iSubsetW_in [[ buffer(11) ]],
  const device int &iSubsetH_in [[ buffer(12) ]],
  const device int &iMaxIteration_in [[ buffer(13) ]],
  const device real_t &fDeltaP_in [[ buffer(14) ]],
  const device uchar *d_tarImg [[ buffer(15) ]],
  const device real_t *whole_d_dInvHessian [[ buffer(16) ]],
  const device real_t4 *m_dTBicubic [[ buffer(17) ]],
  const device real_t2 *whole_d_2dRDescent [[ buffer(18) ]],
  const device real_t *whole_d_dSubsetAveR [[ buffer(19) ]],
  // Tempts.
  device real_t *whole_d_dSubsetT [[ buffer(20) ]],
  device real_t *whole_d_dSubsetAveT [[ buffer(21) ]],
  // Outputs.
  device int *whole_d_iIteration [[ buffer(22) ]],
  device real_t *whole_d_dP [[ buffer(23) ]],
  // Thread params.
  uint2 threadIdx [[ thread_position_in_threadgroup ]],
  uint2 blockDim [[ threads_per_threadgroup ]],
  uint2 blockIdx [[ threadgroup_position_in_grid ]]) {
  
  int iImgWidth = iImgWidth_in;
  int iImgHeight = iImgHeight_in;
  int iStartX = iStartX_in;
  int iStartY = iStartY_in;
  int iROIWidth = iROIWidth_in;
  int iROIHeight = iROIHeight_in;
  int iSubsetX = iSubsetX_in;
  int iSubsetY = iSubsetY_in;
  int iSubsetW = iSubsetW_in;
  int iSubsetH = iSubsetH_in;
  int iMaxIteration = iMaxIteration_in;
  real_t fDeltaP = fDeltaP_in;
  
	threadgroup real_t sm[BLOCK_SIZE_64];
	threadgroup real_t DP[6];
	threadgroup real_t Warp[6];
	threadgroup real_t P[6];
	threadgroup int break_sig[1];

	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;

	real_t fWarpX, fWarpY;
	int iTempX, iTempY;
	real_t fTempX, fTempY, fTempX2, fTempX3, fTempY2, fTempY3;
	real_t ftemptVal;
	int size = iSubsetH*iSubsetW;

	device real_t *fSubsetT = whole_d_dSubsetT + iSubsetH*iSubsetW*bid;
	device real_t *fSubsetAveT = whole_d_dSubsetAveT + (iSubsetH*iSubsetW + 1)*bid;
	const device real_t *fInvHessian = whole_d_dInvHessian + bid * 36;
	const device real_t2 *fRDescent = whole_d_2dRDescent + bid*iSubsetH*iSubsetW * 3;
	const device real_t *fSubsetAveR = whole_d_dSubsetAveR + bid*(iSubsetH*iSubsetW + 1);

	if (tid == 0) {
		// Transfer the initial guess to IC-GN algorithm
		P[0] = d_fU[bid];
		P[1] = 0;
		P[2] = 0;
		P[3] = d_fV[bid];
		P[4] = 0;
		P[5] = 0;

		// Initialize the warp matrix
		Warp[0] = 1 + P[1];
		Warp[1] = P[2];
		Warp[2] = P[0];
		Warp[3] = P[4];
		Warp[4] = 1 + P[5];
		Warp[5] = P[3];
	}
	if (tid == 32) {
		break_sig[0] = 0;
	}
  
	sync_threads();
  
	int iIteration;
	for (iIteration = 0; iIteration < iMaxIteration; iIteration++) {
		real_t mySum = 0;
		for (int id = tid; id < size; id += dim) {
			int l = id / iSubsetW;
			int m = id % iSubsetW;
			if (l < iSubsetH && m < iSubsetW) {
				fWarpX = d_iPOIXY[2 * bid + 1] + Warp[0] * (m - iSubsetX) + Warp[1] * (l - iSubsetY) + Warp[2];
				fWarpY = d_iPOIXY[2 * bid + 0] + Warp[3] * (m - iSubsetX) + Warp[4] * (l - iSubsetY) + Warp[5];

				if (fWarpX < iStartX) fWarpX = iStartX;
				if (fWarpY < iStartY) fWarpY = iStartY;
				if (fWarpX >= iROIWidth + iStartX)  fWarpX = iROIWidth + iStartX - 1;
				if (fWarpY >= iROIHeight + iStartY) fWarpY = iROIHeight + iStartY - 1;

				iTempX = int(fWarpX);
				iTempY = int(fWarpY);

				fTempX = fWarpX - iTempX;
				fTempY = fWarpY - iTempY;
        
				if ((fTempX <= 0.0000001) && (fTempY <= 0.0000001)) {
					ftemptVal = (real_t)d_tarImg[iTempY * iImgWidth + iTempX];
				} else {
					// Unroll for loop
					real_t4 a1, a2, a3, a4;
					a1 = m_dTBicubic[0 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a2 = m_dTBicubic[1 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a3 = m_dTBicubic[2 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a4 = m_dTBicubic[3 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
          
          fTempX2 = fTempX * fTempX;
          fTempX3 = fTempX2 * fTempX;
          
          fTempY2 = fTempY * fTempY;
          fTempY3 = fTempY2 * fTempY;
          
					ftemptVal =
						a1.w +
						a1.x * fTempX +
						a1.y * fTempX2 +
						a1.z * fTempX3 +

						a2.w * fTempY +
						a2.x * fTempY * fTempX +
						a2.y * fTempY * fTempX2 +
						a2.z * fTempY * fTempX3 +

						a3.w * fTempY2 +
						a3.x * fTempY2 * fTempX +
						a3.y * fTempY2 * fTempX2 +
						a3.z * fTempY2 * fTempX3 +

						a4.w * fTempY3 +
						a4.x * fTempY3 * fTempX +
						a4.y * fTempY3 * fTempX2 +
						a4.z * fTempY3 * fTempX3;
				}
        
				fSubsetT[l * iSubsetW + m] = ftemptVal;
				mySum += ftemptVal / size;
			}
		}

		sync_threads();

		real_t avg;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
		sync_threads();
		avg = sm[0];
		mySum = 0;
    
		for (int id = tid; id < size; id += dim) {
			ftemptVal = fSubsetT[id] - avg;
			mySum += ftemptVal * ftemptVal;
			fSubsetAveT[id + 1] = ftemptVal;
		}
    
		sync_threads();
    
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
    
		sync_threads();

		if (tid == 0) {

			fSubsetAveT[0] = ftemptVal = sqrt(sm[tid]);
			sm[tid] = fSubsetAveR[0] / ftemptVal;
		}

		real_t n0, n1, n2, n3, n4, n5;
		n0 = 0; n1 = 0; n2 = 0; n3 = 0; n4 = 0; n5 = 0;
		real_t2 rd;
		sync_threads();
		real_t Nor = sm[0]; // m_dSubNorR[0] / m_dSubNorT[0];
    
		for (int id = tid; id < size; id += dim) {
			ftemptVal = (Nor)* fSubsetAveT[id + 1] - fSubsetAveR[id + 1];
			rd = fRDescent[id];
			n0 += (rd.x * ftemptVal);
			n1 += (rd.y * ftemptVal);
			rd = fRDescent[size + id];
			n2 += (rd.x * ftemptVal);
			n3 += (rd.y * ftemptVal);
			rd = fRDescent[size * 2 + id];
			n4 += (rd.x * ftemptVal);
			n5 += (rd.y * ftemptVal);
		}

		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n0, tid);
    
		if (tid < 6)
			n0 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n1, tid);
    
		if (tid < 6)
			n1 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n2, tid);
    
		if (tid < 6)
			n2 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n3, tid);
    
		if (tid < 6)
			n3 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n4, tid);
    
		if (tid < 6)
			n4 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n5, tid);
    
		if (tid < 6)
			n5 = sm[0];
		if (tid < 6) {
			DP[tid] =
				fInvHessian[tid * 6 + 0] * n0 +
				fInvHessian[tid * 6 + 1] * n1 +
				fInvHessian[tid * 6 + 2] * n2 +
				fInvHessian[tid * 6 + 3] * n3 +
				fInvHessian[tid * 6 + 4] * n4 +
				fInvHessian[tid * 6 + 5] * n5;
		}
		sync_threads();

		if (tid == 0) {
    
			ftemptVal = (1 + DP[1]) * (1 + DP[5]) - DP[2] * DP[4];
			Warp[0] = ((1 + P[1]) * (1 + DP[5]) - P[2] * DP[4]) / ftemptVal;
			Warp[1] = (P[2] * (1 + DP[1]) - (1 + P[1]) * DP[2]) / ftemptVal;
			Warp[2] = P[0] + (P[2] * (DP[0] * DP[4] - DP[3] - DP[3] * DP[1]) - (1 + P[1]) * (DP[0] * DP[5] + DP[0] - DP[2] * DP[3])) / ftemptVal;
			Warp[3] = (P[4] * (1 + DP[5]) - (1 + P[5]) * DP[4]) / ftemptVal;
			Warp[4] = ((1 + P[5]) * (1 + DP[1]) - P[4] * DP[2]) / ftemptVal;
			Warp[5] = P[3] + ((1 + P[5]) * (DP[0] * DP[4] - DP[3] - DP[3] * DP[1]) - P[4] * (DP[0] * DP[5] + DP[0] - DP[2] * DP[3])) / ftemptVal;

			// Update DeltaP
			P[0] = Warp[2];
			P[1] = Warp[0] - 1;
			P[2] = Warp[1];
			P[3] = Warp[5];
			P[4] = Warp[3];
			P[5] = Warp[4] - 1;

			if (sqrt(
				sqr(DP[0]) +
				sqr(DP[1] * iSubsetX) +
				sqr(DP[2] * iSubsetY) +
				sqr(DP[3]) +
				sqr(DP[4] * iSubsetX) +
				sqr(DP[5] * iSubsetY))
				< fDeltaP) {
        
				break_sig[0] = 1;
			}
		}
		sync_threads();
		if (break_sig[0] == 1) break;
	}
	if (tid == 0) {
		whole_d_iIteration[bid] = iIteration;
		d_fV[bid] = P[3];
		d_fU[bid] = P[0];
	}
	if (tid < 6) {
		whole_d_dP[bid * 6 + tid] = P[tid];
	}
}
