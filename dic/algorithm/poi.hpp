//
//  POI.hpp
//  dic
//
//  Created by Tue Le on 9/11/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef POI_hpp
#define POI_hpp

#include <stdio.h>

struct POI {
  int X;
  int Y;
  double dP0[6];
  double dP[6];
  double dDP[6];
  double ZNCC;
  double ZNSSD;
  int iDarkSubset;
  int iOutofROI;
  int iInvertibleMatrix;
  int iMaxIteration;
  int iIteration;
  double dConvergeCriterion;
  int iProcessed;
};

#ifdef __cplusplus
extern "C" {
#endif

typedef struct POI POI;

int poiGetX(POI* poi);
int poiGetY(POI* poi);

double poiGetU(POI* poi);
double poiGetV(POI* poi);

#ifdef __cplusplus
}
#endif

#endif /* POI_hpp */
