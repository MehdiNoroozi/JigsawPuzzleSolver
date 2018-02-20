#! /bin/sh
# Copyright (C) 2009-2014 Free Software Foundation, Inc.
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

# Make sure flavors like 'cygnus', 'gnu', 'gnits' and command line
# options like '--ignore-deps' and '--silent-rules' are preserved across
# automake reruns.

. test-init.sh

cat >> configure.ac << 'END'
AM_MAINTAINER_MODE
AC_OUTPUT
END

: > Makefile.am
: > NEWS
: > README
: > AUTHORS
: > ChangeLog
: > THANKS

$ACLOCAL
$AUTOCONF
# Order flavors so that all needed files are installed early.
for flavor in --gnits --gnu --foreign --ignore-deps; do

  $AUTOMAKE --add-missing $flavor
  ./configure --enable-maintainer-mode
  grep " $flavor" Makefile
  $MAKE

  # Two code paths in configure.am:
  # - either a file in $(am__configure_deps) has been updated ...
  $sleep
  touch aclocal.m4
  $MAKE
  grep " $flavor" Makefile

  # - ... or not; i.e., Makefile.am or an included file has.
  $sleep
  touch Makefile.am
  $MAKE
  grep " $flavor" Makefile

done

:
