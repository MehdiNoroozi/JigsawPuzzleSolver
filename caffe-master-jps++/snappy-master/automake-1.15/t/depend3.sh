#! /bin/sh
# Copyright (C) 1997-2014 Free Software Foundation, Inc.
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

# Test _DEPENDENCIES variable.  From Lee Iverson.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_SUBST(DEPS)
END

cat > Makefile.am << 'END'
bin_PROGRAMS = TerraVision

TerraVision_SOURCES = \
	AboutDialog.c Clock.c Dialogs.c DrawModel.c \
	TsmWidget.c Gats.c GATSDialogs.c Model.c ModelAnim.c \
	ScannedMap.c \
        TerraVision.c TerraVisionAvs.c TerraVisionCAVE.c \
	Texture.c ThreeDControl.c ThreeDPanel.c \
	ThreeDWidget.c ThreeDWidget1.c TileManager.c \
	TileRequester.c TwoDWidget.c \
        Visible.c RequestGenerator.c X11FrameGrab.c \
	matrix.c pixmaps.c xpmhash.c xpmread.c xcolor.c xv24to8.c

DEPS = @DEPS@

TerraVision_DEPENDENCIES = $(DEPS)
END

$ACLOCAL
$AUTOMAKE
