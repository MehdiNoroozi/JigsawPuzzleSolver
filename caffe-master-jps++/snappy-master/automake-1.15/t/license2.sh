#! /bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Check that installing 'COPYING' outputs a warning.

. test-init.sh

cat > Makefile.am << 'END'
AUTOMAKE_OPTIONS = gnu
END

: >AUTHORS
: >NEWS
: >README
: >ChangeLog
: >INSTALL

$ACLOCAL
AUTOMAKE_fails
grep 'COPYING' stderr

AUTOMAKE_run --add-missing
grep 'COPYING' stderr
grep 'GNU General Public License' stderr
grep 'Consider adding.*version control' stderr
test -f COPYING
