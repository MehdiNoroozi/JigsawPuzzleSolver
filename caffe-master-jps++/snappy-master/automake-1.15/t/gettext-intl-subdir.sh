#! /bin/sh
# Copyright (C) 2006-2014 Free Software Foundation, Inc.
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

# Check gettext 'AM_GNU_GETTEXT_INTL_SUBDIR' support.

required='gettext'
. test-init.sh

cat >>configure.ac <<END
AM_GNU_GETTEXT([external])
AM_GNU_GETTEXT_INTL_SUBDIR
AC_OUTPUT
END

echo 'SUBDIRS = po' >Makefile.am
mkdir po

# If aclocal fails, assume the gettext macros are too old and do not
# define AM_GNU_GETTEXT_INTL_SUBDIR.
$ACLOCAL || skip_ "your gettext macros are probably too old"

# config.rpath is required.
: >config.rpath

# intl/ is required.
AUTOMAKE_fails --add-missing
grep 'AM_GNU_GETTEXT.*intl.*SUBDIRS' stderr

mkdir intl
AUTOMAKE_fails --add-missing
grep 'AM_GNU_GETTEXT.*intl.*SUBDIRS' stderr

echo 'SUBDIRS = po intl' > Makefile.am
$AUTOMAKE --add-missing

:
