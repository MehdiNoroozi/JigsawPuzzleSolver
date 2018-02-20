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

# Make sure Automake will install several copies of required files if needed.
# Reported by Marius Vollmer.

. test-init.sh

cat >> configure.ac <<EOF
AC_CONFIG_FILES([one/Makefile two/Makefile])
AC_OUTPUT
EOF

mkdir one
mkdir two

echo 'SUBDIRS = one two' > Makefile.am
echo 'info_TEXINFOS = mumble.texi' > one/Makefile.am
cat >one/mumble.texi <<'END'
@setfilename mumble.info
@include version.texi
END

cp one/Makefile.am one/mumble.texi two

$ACLOCAL
$AUTOMAKE --add-missing --copy

test -f one/mdate-sh
test -f one/texinfo.tex
test -f two/mdate-sh
test -f two/texinfo.tex

:
