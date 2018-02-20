#! /bin/sh
# Copyright (C) 2005-2014 Free Software Foundation, Inc.
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

# Calling AM_AUTOMAKE_VERSION by hand is a bug.

. test-init.sh

echo 'AM_AUTOMAKE_VERSION([1.9])' >>configure.ac
$ACLOCAL 2>stderr && { cat stderr >&2; exit 0; }
cat stderr >&2
$FGREP 'AM_INIT_AUTOMAKE([1.9])' stderr
