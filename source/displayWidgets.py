"""
Created on Fri Dec 20 10:55:10 2013

@author: erlean
"""
from PyQt4 import QtCore, QtGui, QtSvg
from PyQt4 import Qwt5 as Qwt
import numpy as np
import dicom
from qtutils import qStringToStr
import imageTools
from qtutils import mkColor, mkPen, mkBrush
import resources
from dataImporter import validTags

legendColor = mkColor(255, 255, 255, 0)


class TableWidget(QtGui.QTableWidget):
    def __init__(self, parent=None, strech=True, data=None):
        super(TableWidget, self).__init__(parent)
        self.setAcceptDrops(False)
        self.setColumnCount(1)
        self.setRowCount(1)
        self.setSortingEnabled(False)
        self.setEditTriggers(self.NoEditTriggers)
        self.setDragEnabled(True)
        self.setDragDropMode(QtGui.QAbstractItemView.DragOnly)
        if strech:
            self.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
            self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        else:
            self.verticalHeader().setStretchLastSection(True)
            self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
#        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum))
        self.setMaximumHeight(self.minimumSizeHint().height()+10)
        if data is not None:
            self.setData(data)

    def setData(self, dat):
        nr = len(dat)
        nc = max([len(c) for c in dat])
        self.setColumnCount(nc)
        self.setRowCount(nr)
        for c in range(nc):
            for r in range(nr):
                self.setItem(r, c, QtGui.QTableWidgetItem(""))
        i = 0
        j = 0
        for r in dat:
            for c in r:
                self.item(j, i).setText(str(c))
                i += 1
            i = 0
            j += 1
        self.sizeHint()
        self.setMaximumHeight(self.minimumSizeHint().height() + 10)
#        self.setMinimumHeight(self.minimumSizeHint().height() + 10)

    @QtCore.pyqtSlot()
    def copyToClipboard(self):
        self.selectAll()
        QtGui.QApplication.clipboard().setMimeData(self.mimeData([None]))
        self.clearSelection()

    def mimeTypes(self):
        return QtCore.QStringList(['text/plain', 'text/html', ])

    def startDrag(self, action):
        super(TableWidget, self).startDrag(QtCore.Qt.CopyAction)

    def mimeData(self, items):
        if len(items) == 0:
            return 0
        txt = QtCore.QString("")
        selRanges = self.selectedRanges()
        rrange = range(min([ran.topRow() for ran in selRanges]),
                       max([ran.bottomRow() for ran in selRanges]) + 1)
        crange = range(min([ran.leftColumn() for ran in selRanges]),
                       max([ran.rightColumn() for ran in selRanges]) + 1)
        for r in rrange:
            for c in crange:
                if self.item(r, c).isSelected():
                    txt.append(self.item(r, c).text())
                    if c < crange[-1]:
                        txt.append("\t")
            if r < rrange[-1]:
                txt.append("\n")
        m = QtCore.QMimeData()
        m.setText(txt)
        html = QtCore.Qt.escape(txt)
        html.replace("\t", "<td>")
        html.replace("\n", "\n<tr><td>")
        html.prepend("<table>\n<tr><td>")
        html.append("\n</table>")
        m.setHtml(html)
        return m

    def mouseReleaseEvent(self, e):
        e.setAccepted(False)

#    def sizeHint(self):
#        h = self.columnCount() * 30
#        v = self.rowCount() * 50
#        return QtCore.QSize(h, v)


class ErrorBarPlotCurve(Qwt.QwtPlotCurve):

    def __init__(self,
                 title=QtCore.QString(),
                 x=[], y=[], dx=None, dy=None,
                 curvePen=QtGui.QPen(QtCore.Qt.NoPen),
                 curveStyle=Qwt.QwtPlotCurve.NoCurve,
                 curveSymbol=Qwt.QwtSymbol(),
                 errorPen=QtGui.QPen(QtCore.Qt.NoPen),
                 errorCap=0,
                 errorOnTop=False,
                 ):
        """A curve of x versus y data with error bars in dx and dy.

        Horizontal error bars are plotted if dx is not None.
        Vertical error bars are plotted if dy is not None.

        x and y must be sequences with a shape (N,) and dx and dy must be
        sequences (if not None) with a shape (), (N,), or (2, N):
        - if dx or dy has a shape () or (N,), the error bars are given by
          (x-dx, x+dx) or (y-dy, y+dy),
        - if dx or dy has a shape (2, N), the error bars are given by
          (x-dx[0], x+dx[1]) or (y-dy[0], y+dy[1]).

        curvePen is the pen used to plot the curve

        curveStyle is the style used to plot the curve

        curveSymbol is the symbol used to plot the symbols

        errorPen is the pen used to plot the error bars

        errorCap is the size of the error bar caps

        errorOnTop is a boolean:
        - if True, plot the error bars on top of the curve,
        - if False, plot the curve on top of the error bars.
        """
        super(ErrorBarPlotCurve, self).__init__(title)
#        Qwt.QwtPlotCurve.__init__(self)
        self.setData(x, y, dx, dy)
        self.setPen(curvePen)
        self.setStyle(curveStyle)
        self.setSymbol(curveSymbol)
        self.errorPen = errorPen
        self.errorCap = errorCap
        self.errorOnTop = errorOnTop

    # __init__()

    def setData(self, x, y, dx=None, dy=None):
        """Set x versus y data with error bars in dx and dy.

        Horizontal error bars are plotted if dx is not None.
        Vertical error bars are plotted if dy is not None.

        x and y must be sequences with a shape (N,) and dx and dy must be
        sequences (if not None) with a shape (), (N,), or (2, N):
        - if dx or dy has a shape () or (N,), the error bars are given by
          (x-dx, x+dx) or (y-dy, y+dy),
        - if dx or dy has a shape (2, N), the error bars are given by
          (x-dx[0], x+dx[1]) or (y-dy[0], y+dy[1]).
        """

        self.__x = np.asarray(x, np.float)
        if len(self.__x.shape) != 1:
            raise RuntimeError('len(asarray(x).shape) != 1')

        self.__y = np.asarray(y, np.float)
        if len(self.__y.shape) != 1:
            raise RuntimeError('len(asarray(y).shape) != 1')
        if len(self.__x) != len(self.__y):
            raise RuntimeError('len(asarray(x)) != len(asarray(y))')

        if dx is None:
            self.__dx = None
        else:
            self.__dx = np.asarray(dx, np.float)
            if len(self.__dx.shape) not in [0, 1, 2]:
                raise RuntimeError('len(asarray(dx).shape) not in [0, 1, 2]')

        if dy is None:
            self.__dy = dy
        else:
            self.__dy = np.asarray(dy, np.float)
            if len(self.__dy.shape) not in [0, 1, 2]:
                raise RuntimeError('len(asarray(dy).shape) not in [0, 1, 2]')

        Qwt.QwtPlotCurve.setData(self, self.__x, self.__y)

    # setData()

    def hasDx(self):
        if self.__dx is not None:
            return True
        return False

    def hasDy(self):
        if self.__dy is not None:
            return True
        return False

    def dX(self, i):
        return self.__dx[i]

    def dY(self, i):
        return self.__dy[i]

    def boundingRect(self):
        """Return the bounding rectangle of the data, error bars included.
        """
        if self.__dx is None:
            xmin = min(self.__x)
            xmax = max(self.__x)
        elif len(self.__dx.shape) in [0, 1]:
            xmin = min(self.__x - self.__dx)
            xmax = max(self.__x + self.__dx)
        else:
            xmin = min(self.__x - self.__dx[0])
            xmax = max(self.__x + self.__dx[1])

        if self.__dy is None:
            ymin = min(self.__y)
            ymax = max(self.__y)
        elif len(self.__dy.shape) in [0, 1]:
            ymin = min(self.__y - self.__dy)
            ymax = max(self.__y + self.__dy)
        else:
            ymin = min(self.__y - self.__dy[0])
            ymax = max(self.__y + self.__dy[1])

        return QtCore.QRectF(xmin, ymin, xmax-xmin, ymax-ymin)

    # boundingRect()

    def drawFromTo(self, painter, xMap, yMap, first, last=-1):
        """Draw an interval of the curve, including the error bars

        painter is the QPainter used to draw the curve

        xMap is the Qwt.QwtDiMap used to map x-values to pixels

        yMap is the Qwt.QwtDiMap used to map y-values to pixels

        first is the index of the first data point to draw

        last is the index of the last data point to draw. If last < 0, last
        is transformed to index the last data point
        """

        if last < 0:
            last = self.dataSize() - 1

        if self.errorOnTop:
            Qwt.QwtPlotCurve.drawFromTo(self, painter, xMap, yMap, first, last)

        # draw the error bars
        painter.save()
        painter.setPen(self.errorPen)

        # draw the error bars with caps in the x direction
        if self.__dx is not None:
            # draw the bars
            if len(self.__dx.shape) in [0, 1]:
                xmin = (self.__x - self.__dx)
                xmax = (self.__x + self.__dx)
            else:
                xmin = (self.__x - self.__dx[0])
                xmax = (self.__x + self.__dx[1])
            y = self.__y
            n, i = len(y), 0
            lines = []
            while i < n:
                yi = yMap.transform(y[i])
                lines.append(QtCore.Qt.QLine(xMap.transform(xmin[i]), yi,
                                             xMap.transform(xmax[i]), yi))
                i += 1
            painter.drawLines(lines)
            if self.errorCap > 0:
                # draw the caps
                cap = self.errorCap/2
                n, i, = len(y), 0
                lines = []
                while i < n:
                    yi = yMap.transform(y[i])
                    lines.append(
                        QtCore.QLine(xMap.transform(xmin[i]), yi - cap,
                                     xMap.transform(xmin[i]), yi + cap))
                    lines.append(
                        QtCore.QLine(xMap.transform(xmax[i]), yi - cap,
                                     xMap.transform(xmax[i]), yi + cap))
                    i += 1
            painter.drawLines(lines)

        # draw the error bars with caps in the y direction
        if self.__dy is not None:
            # draw the bars
            if len(self.__dy.shape) in [0, 1]:
                ymin = (self.__y - self.__dy)
                ymax = (self.__y + self.__dy)
            else:
                ymin = (self.__y - self.__dy[0])
                ymax = (self.__y + self.__dy[1])
            x = self.__x
            n, i, = len(x), 0
            lines = []
            while i < n:
                xi = xMap.transform(x[i])
                lines.append(
                    QtCore.QLine(xi, yMap.transform(ymin[i]),
                                 xi, yMap.transform(ymax[i])))
                i += 1
            painter.drawLines(lines)
            # draw the caps
            if self.errorCap > 0:
                cap = self.errorCap/2
                n, i, j = len(x), 0, 0
                lines = []
                while i < n:
                    xi = xMap.transform(x[i])
                    lines.append(
                        QtCore.QLine(xi - cap, yMap.transform(ymin[i]),
                                     xi + cap, yMap.transform(ymin[i])))
                    lines.append(
                        QtCore.QLine(xi - cap, yMap.transform(ymax[i]),
                                     xi + cap, yMap.transform(ymax[i])))
                    i += 1
            painter.drawLines(lines)

        painter.restore()

        if not self.errorOnTop:
            Qwt.QwtPlotCurve.drawFromTo(self, painter, xMap, yMap, first, last)

    # drawFromTo()
# class ErrorBarPlotCurve


class Legend(Qwt.QwtLegend):
    def __init__(self, parent=None):
        super(Legend, self).__init__(parent)
        self.setItemMode(self.ReadOnlyItem)
#        wid = self.contentsWidget()
#        wid.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
#        palette = wid.palette()
#        palette.setColor(palette.Window, legendColor)
#        wid.setPalette(palette)
#        palette = QtGui.QPalette(palette)
#        palette.setColor(palette.Window, legendColor)
#        self.setPalette(palette)

    def paintEvent(self, ev):
        opt = QtGui.QStyleOption()
        opt.initFrom(self)
        opt.palette.setColor(opt.palette.Window, legendColor)
        p = QtGui.QPainter(self)
        self.style().drawPrimitive(QtGui.QStyle.PE_Widget, opt, p, self)


class PlotWidget(Qwt.QwtPlot):
    def __init__(self, title="", parent=None, legendPosition='upperLeft'):
        title = Qwt.QwtText(QtCore.QString(title))
        super(PlotWidget, self).__init__(title, parent)
        self.curves = []
        self.markers = []
        self.setCanvasLineWidth(0)
        canvas = self.canvas()
        canvas.setFocusIndicator(canvas.NoFocusIndicator)
        canvas.setCursor(QtCore.Qt.ArrowCursor)
        legend = Legend(self)
        legend.contentsWidget().layout().setMaxCols(1)
        self.legendPosition = legendPosition
        self.insertLegend(legend, self.ExternalLegend)
        self.moveLegend()

    def plotAll(self, plots):
        """ plots is on the form:
        plots = {'title':{'x':x, 'y':y, 'yerr':yerr, 'dots':False,
                 'size': 1, 'color': qcolor or*see mkcolor*}}
        """
        plots_sort = []
        for title, plot in plots.items():
            plots_sort.append((title, plot))
        plots_sort.sort(key=lambda x: x[0])

        for title, plot in plots_sort:
            x = plot['x']
            y = plot['y']
            size = plot.get('size', 1)
            color = mkColor(plot.get('color', 'b'))

            if plot.get('dots', False):
                size *= 2.
                pen = mkPen(cosmetic=True, color=mkColor(0.), width=.2)
                brush = mkBrush(color)
                symbol = Qwt.QwtSymbol()
                symbol.setStyle(symbol.Ellipse)
                symbol.setPen(pen)
                symbol.setBrush(brush)
                symbol.setSize(size)
                if 'yerr' in plot:
                    yerr = plot['yerr']
                    errorPen = mkPen(cosmetic=True, color=0., width=2)
                    curve = ErrorBarPlotCurve(title=QtCore.QString(title),
                                              x=x, y=y, dy=yerr,
                                              curveSymbol=symbol,
                                              errorPen=errorPen,
                                              errorOnTop=True)
                else:
                    curve = Qwt.QwtPlotCurve(QtCore.QString(title))
                    curve.setStyle(curve.NoCurve)
                    curve.setSymbol(symbol)
                    curve.setData(x, y)
            else:
                pen = mkPen(cosmetic=True, color=color, width=size)
                curve = Qwt.QwtPlotCurve(QtCore.QString(title))
                curve.setData(x, y)
                curve.setPen(pen)

            if 'plotText' in plot:
                if 'plotTextAlignment' in plot:
                    if plot['plotTextAlignment'] == 'upperLeft':
                        al = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
                    elif plot['plotTextAlignment'] == 'upperRight':
                        al = QtCore.Qt.AlignRight | QtCore.Qt.AlignTop
                    elif plot['plotTextAlignment'] == 'lowerRight':
                        al = QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom
                    elif plot['plotTextAlignment'] == 'lowerLeft':
                        al = QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom
                    elif plot['plotTextAlignment'] == 'centerLeft':
                        al = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignLeft
                    elif plot['plotTextAlignment'] == 'centerRight':
                        al = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignRight
                    else:
                        al = QtCore.Qt.AlignCenter
                else:
                    al = QtCore.Qt.AlignCenter

                for x, y, txt in plot['plotText']:
                    marker = Qwt.QwtPlotMarker()
                    marker.setValue(x, y)
                    marker.setLabel(Qwt.QwtText(txt))
                    marker.setLabelAlignment(al)
                    marker.attach(self)
                    self.legend().remove(marker)
                    self.markers.append(marker)

            curve.setRenderHint(curve.RenderAntialiased)
            curve.setPaintAttribute(curve.PaintFiltered)
            curve.attach(self)
            self.curves.append(curve)
            if not plot.get('inLegend', True):
                self.legend().remove(curve)

    def setBestLegendPos(self):
        upperLeft = self.axisWidget(self.yLeft).geometry().topRight()
        lowerLeft = self.axisWidget(self.yLeft).geometry().bottomRight()
        lowerRight = self.axisWidget(self.xBottom).geometry().topRight()
        upperRight = QtCore.QPoint(lowerRight.x(), upperLeft.y())
        pos = [upperLeft, lowerLeft, upperRight, lowerRight]
        lpos = ['upperLeft', 'lowerLeft', 'upperRight', 'lowerRight']
        d = [1e9, 1e9, 1e9, 1e9]
        for ind, p in enumerate(pos):
            for c in self.curves:
                pind, dist = c.closestPoint(p)
                if pind != -1:
                    d[ind] = min([d[ind], dist])
        self.legendPosition = lpos[d.index(max(d))]

    def legendPos(self):
        if self.legendPosition == "upperRight":
            pos = self.axisWidget(self.yRight).geometry().topRight()
            x = self.axisWidget(self.xBottom).geometry().topRight().x()
            y = self.axisWidget(self.yLeft).geometry().topRight().y()
            legendGeo = self.legend().geometry()
            pos = QtCore.QPoint(x - legendGeo.width(), y)
            return pos
        elif self.legendPosition == "lowerRight":
            pos = self.axisWidget(self.xBottom).geometry().topRight()
            legendGeo = self.legend().geometry()
            pos.setY(pos.y() - legendGeo.height())
            pos.setX(pos.x() - legendGeo.width())
            return pos
        elif self.legendPosition == "lowerLeft":
            pos = self.axisWidget(self.yLeft).geometry().bottomRight()
            legendHeight = self.legend().geometry().height()
            pos.setY(pos.y() - legendHeight)
            return pos
        return self.axisWidget(self.yLeft).geometry().topRight()

    def moveLegend(self):
        self.setBestLegendPos()
        self.legend().move(self.legendPos())

    def resizeEvent(self, ev):
        super(PlotWidget, self).resizeEvent(ev)
        self.moveLegend()

    def showEvent(self, ev):
        super(PlotWidget, self).showEvent(ev)
        self.moveLegend()

    @QtCore.pyqtSlot()
    def saveImage(self):
        imageFilter = "Raster Image (*.png *.jpg *.tiff);;"
        imageFilter += "Vector Graphics (*.svg);;Document Mode (*.pdf *.ps)"

        path = QtGui.QFileDialog.getSaveFileName(self, "Save Image",
                                                 "plot.png", imageFilter)
        if path != "":
            if path.toLower().endsWith(".svg"):
                try:
                    svg = QtSvg.QSvgGenerator()
                    svg.setFileName(path)
                    svg.setResolution(300)
                    svg.setSize(QtCore.QSize(1600, 1200))
                    svg.setViewBox(QtCore.QRect(0, 0, 1600, 1200))
                    self.printToVectorDevice(svg)
                except Exception:
                    pass

            elif path.toLower().endsWith(".ps"):
                printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
                printer.setOutputFormat(QtGui.QPrinter.PostScriptFormat)
                printer.setOutputFileName(path)
                printer.setOrientation(QtGui.QPrinter.Landscape)
                printer.setResolution(300)
                self.printToVectorDevice(printer)
            elif path.toLower().endsWith(".pdf"):
                printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
                printer.setOutputFormat(QtGui.QPrinter.PdfFormat)
                printer.setOutputFileName(path)
                printer.setOrientation(QtGui.QPrinter.Landscape)
                printer.setResolution(300)
                self.printToVectorDevice(printer)
            else:
                qim = self.toQImage()
                try:
                    qim.save(path, "PNG")
                except Exception:
                    pass

    def printToVectorDevice(self, device, dpi=300):
        dpmx = device.logicalDpiX()
        dpmy = device.logicalDpiY()
        # making curves larger
        for curve in self.curves:
            pen = curve.pen()
            pen.setWidthF(pen.widthF()*2)
            curve.setPen(pen)

        filt = Qwt.QwtPlotPrintFilter()
        filt.setOptions(filt.PrintTitle)

        rx = float(dpmx) / float(96)
        ry = float(dpmy) / float(96)
        lpos = self.axisWidget(self.yLeft).geometry().topRight()
        transform = QtGui.QTransform.fromTranslate(lpos.x()*rx, lpos.y()*ry)
        transform.scale(rx*1.5, ry*1.5)

        legend = self.legend()

        p = QtGui.QPainter()
        p.begin(device)
        p.setRenderHint(p.Antialiasing, True)
        p.setRenderHint(p.TextAntialiasing, True)
        p.setRenderHint(p.HighQualityAntialiasing, True)
        self.print_(p, QtCore.QRect(0, 0, device.width(), device.height()),
                    filt)
        p.setTransform(transform)
        legend.render(p, flags=QtGui.QWidget.DrawChildren)
        p.end()

        # making curves smaller
        for curve in self.curves:
            pen = curve.pen()
            pen.setWidthF(pen.widthF()/2.)
            curve.setPen(pen)

    def toQImage(self):
        legendItems = self.legend().legendItems()
        esize = QtCore.QSize(1600, 1200)
        qim = QtGui.QImage(esize, QtGui.QImage.Format_RGB32)
        qim.fill(QtGui.QColor(255, 255, 255))
        qim.setDotsPerMeterX(11811)
        qim.setDotsPerMeterY(11811)
        # making curves larger
        for curve in self.curves:
            pen = curve.pen()
            pen.setWidthF(pen.widthF()*2)
            curve.setPen(pen)

        filt = Qwt.QwtPlotPrintFilter()
        filt.setOptions(filt.PrintTitle)

        self.print_(qim, filt)

        borderx = self.axisWidget(self.yLeft).geometry().width() * 1.05
        bordery = self.axisWidget(self.xBottom).geometry().height() * 1.05
        # border correction
        borderx -= (self.axisWidget(self.yLeft).titleHeightForWidth(
            self.canvas().height()) - self.axisWidget(
            self.yLeft).titleHeightForWidth(400))
        bordery -= (self.axisWidget(self.xBottom).titleHeightForWidth(
            self.canvas().width()) - self.axisWidget(
            self.xBottom).titleHeightForWidth(533))

        scale = 3.
        leg_s = QtCore.QSizeF(self.legend().size())
        size_x = 1600. / scale
        size_y = 1200. / scale

        if self.legendPosition == "upperRight":
            leg_rect = QtCore.QRectF(QtCore.QPointF(size_x - leg_s.width(),
                                     0.), leg_s)
        elif self.legendPosition == "lowerRight":
            leg_rect = QtCore.QRectF(QtCore.QPointF(size_x - leg_s.width(),
                                     size_y - leg_s.height() - bordery), leg_s)
        elif self.legendPosition == "lowerLeft":
            leg_rect = QtCore.QRectF(QtCore.QPointF(borderx,
                                     size_y - bordery - leg_s.height()), leg_s)
        else:
            leg_rect = QtCore.QRectF(QtCore.QPointF(borderx, 0), leg_s)

        p = QtGui.QPainter(qim)
        p.setTransform(QtGui.QTransform.fromScale(scale, scale), True)
        self.legend().render(p, leg_rect.topLeft().toPoint(),
                             flags=QtGui.QWidget.DrawChildren)

        # making curves smaller
        for curve in self.curves:
            pen = curve.pen()
            pen.setWidthF(pen.widthF()/2.)
            curve.setPen(pen)

        # making legend pretty again
        for it in self.legend().legendItems():
            if it not in legendItems:
                self.legend().remove(self.legend().find(it))
        self.legend().layoutContents()
        self.moveLegend()
        return qim

    def getData(self):
        d = []
#        for l in self.itemList():
        for l in self.curves:
            x = []
            y = []
            dx = []
            dy = []
            if isinstance(l, ErrorBarPlotCurve):
                hasDx = l.hasDx()
                hasDy = l.hasDy()
            else:
                hasDx = False
                hasDy = False
            for i in range(l.dataSize()):
                x.append(l.data().x(i))
                y.append(l.data().y(i))
                if hasDx:
                    dx.append(l.dX(i))
                if hasDy:
                    dy.append(l.dY(i))
            r = {'title': l.title().text(), 'x': x, 'y': y}
            if hasDy:
                r['y_err'] = dy
            if hasDx:
                r['x_err'] = dx
            d.append(r)
        return d

    def dataToString(self, seperator="\t"):
        d = self.getData()
        txt = QtCore.QString("")
        if len(d) == 0:
            return txt
        # header
        for item in d:
            txt.append(item['title'])
            txt.append(seperator * (len(item) - 1))
        txt.chop(len(seperator) * (len(d[-1]) - 1))
        txt.append("\n")
        header = ['x', 'y', 'x_err', 'y_err']
        for item in d:
            for key in header:
                if key in item:
                    txt.append(key)
                    txt.append(seperator)
        txt.chop(len(seperator))
        txt.append("\n")
        # plots
        i = 0
        imax = max([len(item['x']) for item in d])
        while i < imax:
            for item in d:
                for key in header:
                    if key in item:
                        if i < len(item[key]):
                            txt.append(str(item[key][i]))
                        txt.append(seperator)
            txt.chop(len(seperator))
            txt.append("\n")
            i += 1
        return txt

    @QtCore.pyqtSlot()
    def copyToClipboard(self):
        txt = self.dataToString()
        m = QtCore.QMimeData()
        m.setText(txt)
        QtGui.QApplication.clipboard().setMimeData(m)

    def mousePressEvent(self, ev):
        if ev.buttons() == QtCore.Qt.LeftButton:
            self.mouseDownPos = ev.globalPos()

    def mouseMoveEvent(self, ev):
        if ev.buttons() == QtCore.Qt.LeftButton:
            lenght = (ev.globalPos() - self.mouseDownPos).manhattanLength()
            if lenght > QtGui.QApplication.startDragDistance():
                qim = self.toQImage()
                md = QtCore.QMimeData()
                md.setImageData(qim)
                drag = QtGui.QDrag(self)
                pix = QtGui.QPixmap.fromImage(qim.scaledToWidth(64))
                drag.setPixmap(pix)
                drag.setMimeData(md)
                # initialiserer drops
                drag.exec_(QtCore.Qt.CopyAction)


class ImageItem(QtGui.QGraphicsItem):
    def __init__(self, image=None, level=None, parent=None, shape=None):
        super(ImageItem, self).__init__(parent)
#        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        if image is None:
            if shape is None:
                shape = (512, 512)
            self.image = np.zeros(shape)
        else:
            self.image = image.view(np.ndarray)
        if image is not None and level is None:
            mi = image.min()
            ma = image.max() + 1
            self.level = ((ma - mi) / 2, ) * 2
        elif level is None:
            self.level = (-100, 100)
        else:
            self.level = level

        self.prepareGeometryChange()
        self.qimage = None

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        x, y = self.image.shape
        return QtCore.QRectF(self.x(), self.y(), y, x)

    def setImage(self, image):
        self.image = image.view(np.ndarray)
        self.prepareGeometryChange()
        self.qimage = None
        self.update(self.boundingRect())

    def setLevels(self, level):
        self.level = level
        self.qimage = None
        self.update(self.boundingRect())

    def render(self):
        self.qimage = imageTools.arrayToQImage(self.image, self.level[0],
                                               self.level[1])

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter, style, widget=None):
        if self.qimage is None:
            self.render()
        painter.drawImage(QtCore.QPointF(self.pos()), self.qimage)
#        super(ImageItem, self).paint(painter, style, widget)


class TextItem(QtGui.QGraphicsSimpleTextItem):
    def __init__(self, *args, **kwargs):
        super(TextItem, self).__init__(*args, **kwargs)
        self.setFlag(self.ItemIgnoresTransformations)

    def setPen(self, pen):
        b = self.brush()
        b.setColor(pen.color())
        self.setBrush(b)


class ImageView(QtGui.QGraphicsView):
    def __init__(self, parent=None, images=None, pos=None):
        super(ImageView, self).__init__(parent)
        self.setAcceptDrops(False)
        self.setScene(QtGui.QGraphicsScene())
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.image = ImageItem()
        self.scene().addItem(self.image)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.SmoothPixmapTransform |
                            QtGui.QPainter.TextAntialiasing)

        self.level = [0, 500]
        self.levelItem = QtGui.QGraphicsSimpleTextItem(" ")
        self.posItem = QtGui.QGraphicsSimpleTextItem()
        self.posItem.setY(self.levelItem.boundingRect().height())
        self.graphItems = []
        self.txtItems = [self.levelItem, self.posItem]
        self.scene().addItem(self.levelItem)
        self.scene().addItem(self.posItem)
        self.mouseDownPos = QtCore.QPoint(0, 0)
        self.imageListIndex = 0
        self.imageList = []
        self.posList = []

        if images is not None:
            self.setNumpyArrays(images, pos)

    def addGraphItem(self, item, title="", pen=None, inside=True):
        if pen is None:
            pen = QtGui.QPen(QtGui.QBrush(QtCore.Qt.black), 1.5)
#            pg.mkPen(cosmetic=False, width=1.5, color=0.0)
        item.setPen(pen)
        self.graphItems.append(item)
        self.scene().addItem(self.graphItems[-1])
        if title != "":
            txtItem = QtGui.QGraphicsSimpleTextItem(title)
            itemRect = item.boundingRect()
            txtRect = txtItem.boundingRect()
            # Testing direction
            if itemRect.height() > itemRect.width() + 1:
                txtItem.setRotation(-90)
                if inside:
                    if itemRect.height() > txtRect.width() and itemRect.width() > txtRect.height():
                        p = itemRect.center()
                        offset = QtCore.QPointF(txtRect.height() / 2.,
                                                -txtRect.width() / 2.)
                        txtItem.setPos(p - offset)
                    else:
                        p = itemRect.center()
                        p += QtCore.QPointF(itemRect.width() / 2.,
                                            txtRect.width() / 2.)
                        txtItem.setPos(p)
                else:
                    p = itemRect.center()
                    p += QtCore.QPointF(itemRect.width() / 2.,
                                        txtRect.width() / 2.)
                    txtItem.setPos(p)

            else:
                if inside:
                    if itemRect.width() >= txtRect.width() and itemRect.height() >= txtRect.height():
                        p = itemRect.center() - txtRect.center()
                        txtItem.setPos(p)
                    else:
                        p = itemRect.center()
                        p += QtCore.QPointF(-txtRect.width() / 2.,
                                            itemRect.height() / 2.)
                        txtItem.setPos(p)
                else:
                    p = itemRect.center()
                    p += QtCore.QPointF(-txtRect.width() / 2.,
                                        itemRect.height() / 2.)
                    txtItem.setPos(p)
#                    if itemRect.height() >= txtRect.height():
#                        p = itemRect.center() - txtRect.center()
#                        txtItem.setPos(p)
#                    else:
#                        p = itemRect.center()
#                        p += QtCore.QPointF(-txtRect.width() / 2.,
#                                            itemRect.height() / 2.)
#                        txtItem.setPos(p)
#                else:
#                    p = itemRect.center()
#                    p += QtCore.QPointF(-txtRect.width() / 2.,
#                                        itemRect.height() / 2.)
#                    txtItem.setPos(p)
#                txtItem.setY(itemRect.bottom())
#                x = itemRect.left() + itemRect.width() / 2.
#                x -= txtRect.width() / 2.
#                txtItem.setX(x)
            self.txtItems.append(txtItem)
            self.scene().addItem(txtItem)

    def updateItemsPen(self):
        qim = self.image.qImage()
        for item in self.graphItems:
            center = item.boundingRect().center().toPoint()
            pen = item.pen()
            if QtGui.qGray(qim.pixel(center.x(), center.y())) > 128:
                pen.setColor(QtCore.Qt.black)
            else:
                pen.setColor(QtCore.Qt.white)
            item.setPen(pen)

        for item in self.txtItems:
            pos = item.pos().toPoint()
            x = pos.x()
            y = pos.y()
            if QtGui.qGray(qim.pixel(x, y)) > 128:
                item.setBrush(QtGui.QBrush(QtCore.Qt.black))
            else:
                item.setBrush(QtGui.QBrush(QtCore.Qt.white))

    def setNumpyArray(self, index):
        self.image.setImage(self.imageList[index])
        self.posItem.setText(self.posList[index])
        self.image.setLevels(self.level)
        self.levelItem.setText('WC ' + str(self.level[0]) +
                               ', WW '+str(self.level[1]))
        self.updateItemsPen()

    def setNumpyArrays(self, arrList, pos=None):
        self.imageList = arrList
        if pos is None:
            self.posList = [''] * len(arrList)
        else:
            self.posList = [str(p) for p in pos]
        self.setNumpyArray(0)

    def setDicoms(self, dcList):
        self.imageList = [imageTools.pixelArray(dc) for dc in dcList]
        self.posList = [str(dc[0x20, 0x32].value[2]) for dc in dcList]
        self.setNumpyArray(0)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.RightButton:
            d = self.mouseDownPos - e.globalPos()
            if d.manhattanLength() > QtGui.QApplication.startDragDistance():
                self.mouseDownPos = e.globalPos()
                self.level[0] += d.y()
                self.level[1] -= d.x()
                if self.level[1] < 1:
                    self.level[1] = 1
                self.image.setLevels(self.level)
                self.levelItem.setText('WC ' + str(self.level[0]) +
                                       ', WW '+str(self.level[1]))
                self.updateItemsPen()
                e.accept()
            return
        elif e.buttons() == QtCore.Qt.LeftButton:
            dist = self.mouseDownPos - e.globalPos()
            if dist.manhattanLength() > QtGui.QApplication.startDragDistance():
                e.accept()
                drag = QtGui.QDrag(self)
                # lager mimedata
                qim = self.toQImage()
                md = QtCore.QMimeData()
                md.setImageData(qim)
                drag.setMimeData(md)
                pix = QtGui.QPixmap.fromImage(qim.scaledToWidth(64))
                drag.setPixmap(pix)
                # initialiserer drops
                drag.exec_(QtCore.Qt.CopyAction)

    def mousePressEvent(self, e):
        self.mouseDownPos = e.globalPos()
        e.setAccepted(False)

    def wheelEvent(self, e):
        if e.modifiers() == QtCore.Qt.ControlModifier:
            dd = (1. + e.delta() / 2200.0)
            self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
            m_pos = self.mapToScene(e.pos())
#            scene_pos = self.sceneRect().center()
            self.scale(dd, dd)

            self.centerOn(m_pos)

        else:
            n_images = len(self.imageList)
            if n_images <= 1:
                return
            self.imageListIndex += (e.delta() > 0)
            self.imageListIndex -= (e.delta() < 0)
            self.imageListIndex %= n_images
            self.setNumpyArray(self.imageListIndex)

    def resizeEvent(self, e):
        super(ImageView, self).resizeEvent(e)
        r = self.sceneRect()
        self.fitInView(r, QtCore.Qt.KeepAspectRatio)

    def mouseReleaseEvent(self, e):
        e.setAccepted(False)

    def toQImage(self):
        rect = self.image.boundingRect()
        h = rect.height()
        w = rect.width()
        b = max([h, w])
        if b >= 1024:
            scale = 1
        else:
            scale = 1024/b
        qim = QtGui.QImage(int(w * scale), int(h * scale),
                           QtGui.QImage.Format_ARGB32_Premultiplied)
        qim.fill(0)
        painter = QtGui.QPainter()
        painter.begin(qim)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
        self.scene().render(painter, source=rect)
        painter.end()
        return qim

    @QtCore.pyqtSlot()
    def saveImage(self):
        imageFilter = "Raster Image (*.png *.jpg *.tiff)"

        path = QtGui.QFileDialog.getSaveFileName(self, "Save Image",
                                                 "image.png", imageFilter)
        if path != "":
            qim = self.toQImage()
            try:
                qim.save(path, "PNG")
            except Exception:
                pass


class ToolBar(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ToolBar, self).__init__(parent)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setMaximumHeight(20)
        self.layout().addItem(QtGui.QSpacerItem(22, 1))

    def addAction(self, icon, name):
        butt = QtGui.QPushButton(self)
        self.layout().addWidget(butt)
        butt.setFlat(True)
        butt.setIcon(icon)
        butt.setIconSize(QtCore.QSize(16, 16))
        butt.setToolTip(name)
        return butt


class BaseWidget(QtGui.QWidget):
    def __init__(self, analysis, parent=None):
        super(BaseWidget, self).__init__(parent)

        self.analysis_success = analysis.success
        self.display_message = analysis.message

        hasImage = False
        hasTable = False
        hasPlot = False

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.setContentsMargins(0, 0, 0, 0)

        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal, self)
        splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical, self)
        layout.addWidget(splitter)

        if len(analysis.images) > 0:
            imV = ImageView(images=analysis.images['images'],
                            pos=analysis.images['pos'], parent=None)
            splitter2.addWidget(imV)
            if len(analysis.graphicsItems) > 0:
                for key, value in analysis.graphicsItems.items():
                    if value[0] == 'circle':
                        item = QtGui.QGraphicsEllipseItem(*value[1:])
                    elif value[0] == 'line':
                        item = QtGui.QGraphicsLineItem(*value[1:])
                    elif value[0] == 'rect':
                        item = QtGui.QGraphicsRectItem(*value[1:])
                    in_label = analysis.graphicsItemsLabelInside.get(key, True)
                    imV.addGraphItem(item, title=key, inside=in_label)
                imV.updateItemsPen()
            hasImage = True
        if len(analysis.dataTable) > 0:
            table = TableWidget(parent=None, data=analysis.dataTable)
            splitter2.addWidget(table)
            hasTable = True
        splitter.addWidget(splitter2)

        if len(analysis.plots) > 0:
            plot = PlotWidget(legendPosition=analysis.plotLegendPosition)
            for key, value in analysis.plotLabels.items():
                plot.setAxisTitle(key, QtCore.QString(value))
                axLim = analysis.plotLimits.get(key, None)
                if axLim is not None:
                    plot.setAxisScale(key, axLim[0], axLim[1])
            plot.plotAll(analysis.plots)
            splitter.addWidget(plot)
            hasPlot = True

        if analysis.warning is not "":
            label = QtGui.QLabel(QtCore.QString(analysis.warning))
            mainLayout.addWidget(label)
            mainLayout.setStretchFactor(label, 1)

        mainLayout.addLayout(layout)
        mainLayout.setStretchFactor(layout, 100)
        self.setLayout(mainLayout)

        # setting up menu
        if any([hasTable, hasImage, hasPlot]):
            buttonLayout = QtGui.QHBoxLayout()
            buttonLayout.setContentsMargins(0, 0, 0, 0)
            toolbar = ToolBar(self)
            buttonLayout.addWidget(toolbar)
            buttonLayout.addItem(QtGui.QSpacerItem(0, 0,
                                 hPolicy=QtGui.QSizePolicy.Expanding))
            mainLayout.insertLayout(0, buttonLayout)

            if hasTable:
                icon = QtGui.QIcon(':Icons/Icons/saveTable.png')
                copyTableAction = toolbar.addAction(icon, 'Copy Table')
                copyTableAction.clicked.connect(table.copyToClipboard)
            if hasImage:
                icon = QtGui.QIcon(':Icons/Icons/saveImage.png')
                saveImageAction = toolbar.addAction(icon, 'Save Image')
                saveImageAction.clicked.connect(imV.saveImage)
            if hasPlot:
                icon = QtGui.QIcon(':Icons/Icons/savePlot.png')
                savePlotAction = toolbar.addAction(icon, 'Save Plot')
                savePlotAction.clicked.connect(plot.saveImage)
                icon = QtGui.QIcon(':Icons/Icons/exportPlot.png')
                copyPlotAction = toolbar.addAction(icon, 'Copy Plot Data')
                copyPlotAction.clicked.connect(plot.copyToClipboard)

    def mouseReleaseEvent(self, e):
        e.setAccepted(False)

    def paintEvent(self, ev):
        super(BaseWidget, self).paintEvent(ev)
        if not self.analysis_success:
            p = QtGui.QPainter(self)
            r = self.rect()
            p.drawText(r, QtCore.Qt.AlignCenter, self.display_message)


class DicomViewImporter(QtCore.QThread):
    image = QtCore.pyqtSignal(np.ndarray, QtCore.QString)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DicomViewImporter, self).__init__(parent)
        self.mustStop = False

    def run(self):
        for p in self.paths:
            if self.mustStop:
                break
            try:
                dc = dicom.read_file(qStringToStr(p))
            except dicom.filereader.InvalidDicomError:
                pass
            else:
                arr = imageTools.pixelArray(dc)
                pos = format(dc[validTags['imageLocation']].value[2], '.3f')
                self.image.emit(arr, QtCore.QString(pos))
        self.finished.emit()

    @QtCore.pyqtSlot()
    def kill(self):
        self.mustStop = True

    @QtCore.pyqtSlot(list)
    def readImages(self, paths):
        self.paths = paths
        self.start()


class DicomView(QtGui.QGraphicsView):
    importPaths = QtCore.pyqtSignal(list)
    killImporter = QtCore.pyqtSignal()

    def __init__(self, paths, parent=None):
        super(DicomView, self).__init__(parent)
        self.setup()
        self.importer = DicomViewImporter()
        self.importer.image.connect(self.loadDicom)
        self.importer.finished.connect(self.importFinished)
        self.killImporter.connect(self.importer.kill)
        self.importPaths.connect(self.importer.readImages)
        self.importPaths.emit(paths)

    def setup(self):
        self.setAcceptDrops(False)
        self.setScene(QtGui.QGraphicsScene())
        self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)

        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.image = ImageItem()
        self.scene().addItem(self.image)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.SmoothPixmapTransform |
                            QtGui.QPainter.TextAntialiasing)

        self.level = [0, 500]
        self.levelItem = QtGui.QGraphicsSimpleTextItem(" ")
        self.posItem = QtGui.QGraphicsSimpleTextItem(" ")
        self.posItem.setY(self.levelItem.boundingRect().height())
        self.loadingItem = QtGui.QGraphicsSimpleTextItem("Loading")
        self.loadingItem.setY(self.levelItem.boundingRect().height() * 2)
        self.graphItems = []
        self.txtItems = [self.levelItem, self.posItem, self.loadingItem]
        for item in self.txtItems:
            item.setFlag(item.ItemIgnoresTransformations)
        self.scene().addItem(self.levelItem)
        self.scene().addItem(self.posItem)
        self.mouseDownPos = QtCore.QPoint(0, 0)
        self.imageListIndex = 0
        self.imageList = []
        self.posList = []

    def updateItemsPen(self):
        qim = self.image.qImage()
        for item in self.graphItems:
            center = item.boundingRect().center().toPoint()
            pen = item.pen()
            if QtGui.qGray(qim.pixel(center.x(), center.y())) > 128:
                pen.setColor(QtCore.Qt.black)
            else:
                pen.setColor(QtCore.Qt.white)
            item.setPen(pen)

        for item in self.txtItems:
            pos = item.pos().toPoint()
            x = pos.x()
            y = pos.y()
            if QtGui.qGray(qim.pixel(x, y)) > 128:
                item.setBrush(QtGui.QBrush(QtCore.Qt.black))
            else:
                item.setBrush(QtGui.QBrush(QtCore.Qt.white))

    def setNumpyArray(self, index):
        self.image.setImage(self.imageList[index])
        self.posItem.setText(self.posList[index])
        self.image.setLevels(self.level)
        self.levelItem.setText('WC ' + str(self.level[0]) +
                               ', WW '+str(self.level[1]))
        self.setSceneRect(self.image.boundingRect())
        self.updateItemsPen()
#        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    @QtCore.pyqtSlot()
    def importFinished(self):
        if len(self.imageList) == 0:
            self.loadingItem.setText("Sorry, no images found...")
        else:
            self.loadingItem.setText("")
            lists = zip(self.imageList, self.posList)
            lists.sort(key=lambda x: float(x[1]))
            self.imageList, self.posList = zip(*lists)
            self.setNumpyArray(0)

    @QtCore.pyqtSlot(np.ndarray, QtCore.QString)
    def loadDicom(self, arr, pos):
        self.imageList.append(arr)
        self.posList.append(pos)
        self.setNumpyArray(-1)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.RightButton:
            d = self.mouseDownPos - e.globalPos()
            if d.manhattanLength() > QtGui.QApplication.startDragDistance():
                self.mouseDownPos = e.globalPos()
                self.level[0] += d.y()
                self.level[1] -= d.x()
                if self.level[1] < 1:
                    self.level[1] = 1
                self.image.setLevels(self.level)
                self.levelItem.setText('WC ' + str(self.level[0]) +
                                       ', WW '+str(self.level[1]))
                self.updateItemsPen()
                e.accept()
            return
        elif e.buttons() == QtCore.Qt.LeftButton:
            dist = self.mouseDownPos - e.globalPos()
            if dist.manhattanLength() > QtGui.QApplication.startDragDistance():
                e.accept()
                drag = QtGui.QDrag(self)
                # lager mimedata
                qim = self.toQImage()
                md = QtCore.QMimeData()
                md.setImageData(qim)
                drag.setMimeData(md)
                pix = QtGui.QPixmap.fromImage(qim.scaledToWidth(64))
                drag.setPixmap(pix)
                # initialiserer drops
                drag.exec_(QtCore.Qt.CopyAction)

    def mousePressEvent(self, e):
        self.mouseDownPos = e.globalPos()
        e.setAccepted(False)

    def wheelEvent(self, e):
        if e.modifiers() == QtCore.Qt.ControlModifier:
            dd = (1. + e.delta() / 2200.0)
            self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
            m_pos = self.mapToScene(e.pos())
            self.scale(dd, dd)

            self.centerOn(m_pos)

        else:
            n_images = len(self.imageList)
            if n_images <= 1:
                return
            self.imageListIndex += (e.delta() > 0)
            self.imageListIndex -= (e.delta() < 0)
            self.imageListIndex %= n_images
            self.setNumpyArray(self.imageListIndex)
        e.accept()

    def showEvent(self, e):
        super(DicomView, self).showEvent(e)
        r = self.sceneRect()
        self.fitInView(r, QtCore.Qt.KeepAspectRatio)

    def mouseReleaseEvent(self, e):
        e.setAccepted(False)

    def toQImage(self):
        rect = self.image.boundingRect()
        h = rect.height()
        w = rect.width()
        b = max([h, w])
        if b >= 1024:
            scale = 1
        else:
            scale = 1024/b
        qim = QtGui.QImage(int(w * scale), int(h * scale),
                           QtGui.QImage.Format_ARGB32_Premultiplied)
        qim.fill(0)
        painter = QtGui.QPainter()
        painter.begin(qim)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
        self.scene().render(painter, source=rect)
        painter.end()
        return qim

    @QtCore.pyqtSlot()
    def saveImage(self):
        imageFilter = "Raster Image (*.png *.jpg *.tiff)"

        path = QtGui.QFileDialog.getSaveFileName(self, "Save Image",
                                                 "plot.png", imageFilter)
        if path != "":
            qim = self.toQImage()
            try:
                qim.save(path, "PNG")
            except Exception:
                pass

    def __del__(self):
        self.killImporter.emit()
        self.imageList = []
        self.posList = []
        self.importer.wait()
