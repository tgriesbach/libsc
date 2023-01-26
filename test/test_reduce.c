/*
  This file is part of the SC Library.
  The SC Library provides support for parallel scientific applications.

  Copyright (C) 2010 The University of Texas System
  Additional copyright (C) 2011 individual authors

  The SC Library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  The SC Library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with the SC Library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA.
*/

#include <sc_reduce.h>

#define NUM_CELLS 512

typedef struct cell
{
  uint64_t            id;
  double              depth;
}
cell_t;

static void
sc_reduce_cell (void *sendbuf, void *recvbuf,
                int sendcount, int itemsize, sc_MPI_Datatype sendtype)
{
  SC_ASSERT (sendtype == sc_MPI_BYTE);
  SC_ASSERT (itemsize == sizeof (cell_t));

  int                 i;
  const cell_t       *cells_send = (cell_t *) sendbuf;
  cell_t             *cells_recv = (cell_t *) recvbuf;

  for (i = 0; i < sendcount; ++i) {
    if (cells_send[i].depth < cells_recv[i].depth) {
      /* copy smaller depth cell to the result */
      cells_recv[i].id = cells_send[i].id;
      cells_recv[i].depth = cells_send[i].depth;
    }
    else if (cells_send[i].depth == cells_recv[i].depth) {
      /* prefer smaller id */
      if (cells_send[i].id < cells_recv[i].depth) {
        cells_recv[i].id = cells_send[i].id;
        cells_recv[i].depth = cells_send[i].depth;
      }
    }
    else {
      /* recvbuf already contains the correct cell */
    }
  }
}

int
main (int argc, char **argv)
{
  int                 mpiret;
  int                 mpirank, mpisize;
  int                 i, j;
  char                cvalue, cresult;
  int                 ivalue, iresult;
  unsigned short      usvalue, usresult;
  long                lvalue, lresult;
  float               fvalue[3], fresult[3], fexpect[3];
  double              dvalue, dresult;
  sc_MPI_Comm         mpicomm;
  cell_t              cells[NUM_CELLS];

  mpiret = sc_MPI_Init (&argc, &argv);
  SC_CHECK_MPI (mpiret);

  mpicomm = sc_MPI_COMM_WORLD;
  mpiret = sc_MPI_Comm_size (mpicomm, &mpisize);
  SC_CHECK_MPI (mpiret);
  mpiret = sc_MPI_Comm_rank (mpicomm, &mpirank);
  SC_CHECK_MPI (mpiret);

  sc_init (mpicomm, 1, 1, NULL, SC_LP_DEFAULT);

  /* test allreduce int max */
  ivalue = mpirank;
  sc_allreduce (&ivalue, &iresult, 1, sc_MPI_INT, sc_MPI_MAX, mpicomm);
  SC_CHECK_ABORT (iresult == mpisize - 1, "Allreduce mismatch");

  /* test reduce float max */
  fvalue[0] = (float) mpirank;
  fexpect[0] = (float) (mpisize - 1);
  fvalue[1] = (float) (mpirank % 9 - 4);
  fexpect[1] = (float) (mpisize >= 9 ? 4 : (mpisize - 1) % 9 - 4);
  fvalue[2] = (float) (mpirank % 6);
  fexpect[2] = (float) (mpisize >= 6 ? 5 : (mpisize - 1) % 6);
  for (i = 0; i < mpisize; ++i) {
    sc_reduce (fvalue, fresult, 3, sc_MPI_FLOAT, sc_MPI_MAX, i, mpicomm);
    if (i == mpirank) {
      for (j = 0; j < 3; ++j) {
        SC_CHECK_ABORTF (fresult[j] == fexpect[j],      /* ok */
                         "Reduce mismatch in %d", j);
      }
    }
  }

  /* test allreduce char min */
  cvalue = (char) (mpirank % 127);
  sc_allreduce (&cvalue, &cresult, 1, sc_MPI_CHAR, sc_MPI_MIN, mpicomm);
  SC_CHECK_ABORT (cresult == 0, "Allreduce mismatch");

  /* test reduce unsigned short min */
  usvalue = (unsigned short) (mpirank % 32767);
  for (i = 0; i < mpisize; ++i) {
    sc_reduce (&usvalue, &usresult, 1, sc_MPI_UNSIGNED_SHORT, sc_MPI_MIN, i,
               mpicomm);
    if (i == mpirank) {
      SC_CHECK_ABORT (usresult == 0, "Reduce mismatch");
    }
  }

  /* test allreduce long sum */
  lvalue = (long) mpirank;
  sc_allreduce (&lvalue, &lresult, 1, sc_MPI_LONG, sc_MPI_SUM, mpicomm);
  SC_CHECK_ABORT (lresult == ((long) (mpisize - 1)) * mpisize / 2,
                  "Allreduce mismatch");

  /* test reduce double sum */
  dvalue = (double) mpirank;
  for (i = 0; i < mpisize; ++i) {
    sc_reduce (&dvalue, &dresult, 1, sc_MPI_DOUBLE, sc_MPI_SUM, i, mpicomm);
    if (i == mpirank) {
      SC_CHECK_ABORT (dresult == ((double) (mpisize - 1)) * mpisize / 2.,       /* ok */
                      "Reduce mismatch");
    }
  }

  /* test reduce_custom_items */
  /* fill array of cell data */
  for (i = 0; i < NUM_CELLS; ++i) {
    cells[i].id = (uint64_t) (NUM_CELLS * mpirank + i);
    cells[i].depth = 1. / ((double) mpirank + 1.);
  }
  /* reduce over the cell data */
  sc_allreduce_custom_items (&cells, &cells, NUM_CELLS, sizeof (cell_t),
                             sc_reduce_cell, mpicomm);
  /* check the result */
  for (i = 0; i < NUM_CELLS; ++i) {
    SC_CHECK_ABORT (cells[i].id == (uint64_t) (NUM_CELLS * (mpisize - 1) + i)
                    && cells[i].depth == 1. / ((double) mpisize),
                    "allreduce_custom_items mismatch");
  }

  sc_finalize ();

  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);

  return 0;
}
