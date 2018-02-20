#!/bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Regression test for a bug reported by Ladislav Strojil.
# Using different extensions for the same language should not
# output the build rules several times.

. test-init.sh

cat >>configure.ac <<EOF
AC_PROG_CXX
EOF

cat >Makefile.am <<EOF
AUTOMAKE_OPTIONS = subdir-objects
bin_PROGRAMS = p q r
p_SOURCES = a.cc b.cpp c.cxx
q_SOURCES = sub/d.cc sub/e.cpp sub/f.cxx
r_SOURCES = g.cc h.cpp i.cxx
r_CXXFLAGS = -DFOO
EOF

$ACLOCAL
$AUTOMAKE

grep '\.o:' Makefile.in > rules
cat rules

# Here is an example of bogus output.  The rules are output several
# times.
#|  .cc.o:
#|  d.o: sub/d.cc
#|  e.o: sub/e.cpp
#|  f.o: sub/f.cxx
#|  r-g.o: g.cc
#|  r-h.o: h.cpp
#|  r-i.o: i.cxx
#|  .cpp.o:
#|  d.o: sub/d.cc
#|  e.o: sub/e.cpp
#|  f.o: sub/f.cxx
#|  r-g.o: g.cc
#|  r-h.o: h.cpp
#|  r-i.o: i.cxx
#|  .cxx.o:
#|  #d.o: sub/d.cc
#|  #e.o: sub/e.cpp
#|  #f.o: sub/f.cxx
#|  #r-g.o: g.cc
#|  #r-h.o: h.cpp
#|  #r-i.o: i.cxx

# Bail out if we find a duplicate.
$PERL -ne 'if (exists $a{$_}) { exit (1) } else { $a{$_} = 1 }' < rules
