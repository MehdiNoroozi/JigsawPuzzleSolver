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

# Test for PR/322.
# Automake 1.6.1 seems to have a problem parsing comments that use
# '\' to span multiple lines.

. test-init.sh

cat >> configure.ac <<'EOF'
AC_OUTPUT
EOF

## There are two tests: one with backslashed comments at the top
## of the file, and one with a rule first.  This is because
## Comments at the top of the file are handled specially
## since Automake 1.5.

cat > Makefile.am << 'EOF'
# SOME_FILES = \
         file1 \
         file2 \
         file3

all-local:
	@echo Good

EOF

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE

grep '# SOME_FILES' Makefile
grep '# *file3' Makefile

cat > Makefile.am << 'EOF'
all-local:
	@echo Good

# SOME_FILES = \
         file1 \
         file2 \
         file3
EOF

$AUTOMAKE
./configure
$MAKE
grep '# SOME_FILES' Makefile
grep '# *file3' Makefile

:
