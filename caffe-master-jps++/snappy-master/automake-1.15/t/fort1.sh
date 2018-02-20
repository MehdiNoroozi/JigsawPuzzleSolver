#! /bin/sh
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

# Test that Automake suggests using AC_PROG_F77/FC if Fortran sources
# are used.

. test-init.sh

cat >Makefile.am <<END
bin_PROGRAMS = hello
hello_SOURCES = hello.f foo.f95
END

$ACLOCAL
AUTOMAKE_fails
grep AC_PROG_F77 stderr
grep AC_PROG_FC stderr
