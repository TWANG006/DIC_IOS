//
//  POI.cpp
//  dic
//
//  Created by Tue Le on 9/11/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#include "POI.hpp"

int poiGetX(POI* poi) {
  return poi->X;
}

int poiGetY(POI* poi) {
  return poi->Y;
}

double poiGetU(POI* poi) {
  return poi->dP[0];
}

double poiGetV(POI* poi) {
  return poi->dP[3];
}
