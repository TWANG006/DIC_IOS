//
//  util.hpp
//  dic
//
//  Created by Tue Le on 9/5/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef util_hpp
#define util_hpp

#include <stdio.h>

template<class T>
T* initialize1D(const int n, const T& v) {
  T* a = new T[n];
  T vv = v;
  for (int i = 0; i < n; i++) {
    a[i] = ++vv;
  }
  return a;
}

template<class T>
T** initialize2D(const int m, const int n, const T& v) {
  T** a = new T*[m];

  for (int i = 0; i < m; i++) {
    a[i] = initialize1D<T>(n, v);
  }

  return a;
}

template<class T>
T*** initialize3D(const int m, const int n, const int k, const T& v) {
  T*** a = new T**[m];

  for (int i = 0; i < m; i++) {
    a[i] = initialize2D<T>(n, k, v);
  }

  return a;
}


template<class T>
T**** initialize4D(const int m, const int n, const int k, const int l, const T& v) {
  T**** a = new T***[m];

  for (int i = 0; i < m; i++) {
    a[i] = initialize3D<T>(n, k, l, v);
  }

  return a;
}

template<class T>
void destroy(T *a) {
  delete []a;
}

template<class T>
void destroy(const int m, T** a) {
  for (int i = 0; i < m; i++) {
    destroy<T>(a[i]);
  }
  delete[] a;
}

template<class T>
void destroy(const int m, const int n, T*** a) {
  for (int i = 0; i < m; i++) {
    destroy<T>(n, a[i]);
  }
  delete[] a;
}

template<class T>
void destroy(const int m, const int n, const int k, T**** a) {
  for (int i = 0; i < m; i++) {
    destroy<T>(n, k, a[i]);
  }
  delete[] a;
}

#endif /* util_hpp */
