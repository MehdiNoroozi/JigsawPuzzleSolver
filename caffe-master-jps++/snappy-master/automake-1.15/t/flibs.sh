#! /bin/sh
# Copyright (C) 1998-2014 Free Software Foundation, Inc.
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

# Make sure 'AC_F77_LIBRARY_LDFLAGS' works properly.
# Matthew D. Langston <langston@SLAC.Stanford.EDU>

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_F77
AC_F77_LIBRARY_LDFLAGS
END

# Tue Aug 11 09:50:48 1998  Matthew D. Langston  <langston@SLAC.Stanford.EDU>
#
# This test currently fails with automake v. 1.3 since automake assumes
# that elements of 'bin_PROGRAMS' (e.g. zardoz) without a corresponding
# '_SOURCES' (e.g. zardoz_SOURCES) should be compiled from 'zardoz.c'
# whether or not 'zardoz.c' actually exists.  For example, even if the
# file 'zardoz.c' doesn't exist but the file 'zardoz.f' does exist, this
# tests would still fail.
#
# Therefore, for now I have put in the line 'zardoz_SOURCES = zardoz.f'
# (see below) so that automake's top-level 'make check' won't fail, but
# this line should be removed once automake handles this situation
# correctly.

cat > Makefile.am <<'END'
bin_PROGRAMS = zardoz
zardoz_SOURCES = zardoz.f
zardoz_LDADD = @FLIBS@
END

: > zardoz.f
: > config.guess
: > config.sub

$ACLOCAL
$AUTOMAKE

grep '@FLIBS@' Makefile.in
