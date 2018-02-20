#! /bin/sh
# Copyright (C) 2001-2014 Free Software Foundation, Inc.
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

# Test to make sure variable is expanded properly.
# From Adam J. Richter.

required=libtool
. test-init.sh

cat >> configure.ac << 'END'
AC_SUBST([LTLIBOBJS])
AM_PROG_AR
AC_PROG_LIBTOOL
END

cat > Makefile.am << 'END'
lib_LTLIBRARIES = libpanel_applet.la
libpanel_applet_la_SOURCES = \
	applet-widget.c
libpanel_applet_la_LDFLAGS = -version-info 0:1:0 -rpath $(libdir)
libpanel_applet_la_LIBADD = -lm
END

: > ltconfig
: > ltmain.sh
: > ar-lib
: > config.guess
: > config.sub

$ACLOCAL
$AUTOMAKE

:
