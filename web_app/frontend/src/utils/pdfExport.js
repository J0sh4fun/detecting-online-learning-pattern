import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

export function exportReportToPdf(report) {
  try {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    let currentY = 20;

    // --- HELPER: NATIVE CHART DRAWING ---
    const drawLineChart = (data, x, y, width, height, title, yMax = 100) => {
      // Background & Grid
      doc.setDrawColor(241, 245, 249);
      doc.setLineWidth(0.1);
      for (let i = 0; i <= 4; i++) {
        const gridY = y + height - (i * height) / 4;
        doc.line(x, gridY, x + width, gridY);
        doc.setFontSize(7);
        doc.setTextColor(148, 163, 184);
        doc.text(`${Math.round((i * yMax) / 4)}`, x - 8, gridY + 2);
      }

      // Title
      doc.setFontSize(10);
      doc.setTextColor(71, 85, 105);
      doc.text(title, x, y - 5);

      // Axes
      doc.setDrawColor(203, 213, 225);
      doc.setLineWidth(0.5);
      doc.line(x, y, x, y + height); // Y
      doc.line(x, y + height, x + width, y + height); // X

      // The Line
      if (data.length > 1) {
        doc.setDrawColor(99, 102, 241); // Indigo-500
        doc.setLineWidth(1);
        for (let i = 0; i < data.length - 1; i++) {
          const x1 = x + (i * width) / (data.length - 1);
          const y1 = y + height - (data[i].value * height) / yMax;
          const x2 = x + ((i + 1) * width) / (data.length - 1);
          const y2 = y + height - (data[i + 1].value * height) / yMax;
          doc.line(x1, y1, x2, y2);
        }
      }
    };

    // --- DATA PREPARATION ---
    const allTimestamps = new Set();
    report.students.forEach(s => s.timeline.forEach(p => allTimestamps.add(new Date(p.timestamp).getTime())));
    const sortedTs = Array.from(allTimestamps).sort();
    const bucketCount = 25;
    const step = Math.max(1, Math.floor(sortedTs.length / bucketCount));
    
    const chartData = [];
    for (let i = 0; i < sortedTs.length; i += step) {
      const ts = sortedTs[i];
      let sumScore = 0, presentCount = 0, camOnCount = 0;
      report.students.forEach(s => {
        const p = s.timeline.find(pt => Math.abs(new Date(pt.timestamp).getTime() - ts) < 5000);
        if (p) { sumScore += p.score; presentCount++; if (p.camera_on) camOnCount++; }
      });
      chartData.push({ avgScore: presentCount > 0 ? sumScore / presentCount : 0, cameraOn: camOnCount, total: presentCount });
    }

    // --- PDF GENERATION ---
    // Header
    doc.setFontSize(24);
    doc.setTextColor(99, 102, 241);
    doc.text('Class Analysis Report', 14, currentY);
    currentY += 10;

    doc.setFontSize(10);
    doc.setTextColor(100, 116, 139);
    doc.text(`${report.room_name} | Generated: ${new Date(report.generated_at).toLocaleString()}`, 14, currentY);
    currentY += 15;

    // Summary Box
    doc.setDrawColor(99, 102, 241);
    doc.setLineWidth(0.5);
    doc.rect(14, currentY, pageWidth - 28, 25);
    
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.text('CLASS AVERAGE', 20, currentY + 10);
    doc.text('TOTAL STUDENTS', 80, currentY + 10);
    doc.text('TEACHER ID', 140, currentY + 10);

    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59);
    doc.text(`${report.class_average_score}%`, 20, currentY + 18);
    doc.text(`${report.students.length}`, 80, currentY + 18);
    doc.text(`${report.teacher_id}`, 140, currentY + 18);
    currentY += 40;

    // Charts (Native Drawing)
    drawLineChart(chartData.map(d => ({ value: d.avgScore })), 20, currentY, 80, 40, 'FOCUS ENGAGEMENT TREND (%)');
    drawLineChart(chartData.map(d => ({ value: d.cameraOn })), 115, currentY, 75, 40, 'ACTIVE CAMERA STREAMS', Math.max(1, report.students.length));
    currentY += 60;

    // Student Table
    doc.setFontSize(14);
    doc.setTextColor(99, 102, 241);
    doc.text('Individual Performance & Behavioral Breakdown', 14, currentY);
    currentY += 5;

    const tableData = report.students.map(s => {
      const counts = {};
      s.timeline.forEach(p => { counts[p.status] = (counts[p.status] || 0) + 1; });
      const total = s.timeline.length || 1;
      const sortedLabels = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      
      return {
        id: s.student_id,
        score: `${Math.round(s.average_score)}%`,
        rating: s.average_score >= 70 ? 'Focused' : s.average_score >= 30 ? 'Distracted' : 'Absence',
        labels: sortedLabels.map(([l, c]) => ({ label: l, percent: Math.round((c / total) * 100) }))
      };
    });

    autoTable(doc, {
      startY: currentY,
      head: [['Student ID', 'Score', 'Rating', 'Behavioral Distribution (Percentage)']],
      body: tableData.map(d => [d.id, d.score, d.rating, '']),
      theme: 'striped',
      headStyles: { fillColor: [99, 102, 241], fontSize: 10 },
      styles: { cellPadding: 6, fontSize: 9, valign: 'middle' },
      columnStyles: {
        3: { cellWidth: 90 }
      },
      didDrawCell: (data) => {
        if (data.section === 'body' && data.column.index === 3) {
          const student = tableData[data.row.index];
          const x = data.cell.x + 2;
          let y = data.cell.y + 4;
          
          student.labels.slice(0, 3).forEach(item => {
            doc.setFontSize(6);
            doc.setTextColor(100, 116, 139);
            doc.text(`${item.label}: ${item.percent}%`, x, y);
            
            // Draw small bar
            const barWidth = 40;
            doc.setDrawColor(226, 232, 240);
            doc.rect(x + 35, y - 2, barWidth, 2);
            doc.setFillColor(99, 102, 241);
            doc.rect(x + 35, y - 2, (barWidth * item.percent) / 100, 2, 'F');
            
            y += 4;
          });
        }
      },
      margin: { left: 14, right: 14 }
    });

    // Footer
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(148, 163, 184);
      doc.text(`AI Classroom Monitoring - Page ${i} of ${totalPages}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
    }

    doc.save(`ClassReport_${report.room_code}.pdf`);
  } catch (error) {
    console.error('PDF Export Error:', error);
    alert('Failed to generate PDF. Check console.');
  }
}
