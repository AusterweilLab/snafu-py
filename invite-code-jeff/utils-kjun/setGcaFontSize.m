function setGcaFontSize(fsize, fontName)
% function setGcaFontSize(fsize)
% DESCRIPTION
%  Change current figure's font size to 'fsize'. Default font as 'times new roman' is used.
%  Type listfonts to see the types of fonts
% INPUT
%  fsize: target font size
%  fontName: (optional) font name
% OUTPUT
%  none
% Example
%  figure; x=1:.1:10; plot(x,exp(x)); setGcaFontSize(13);

if (~exist('fontName', 'var'))
  fontName = 'Helvetica';
end

set(gca, 'FontSize', fsize, 'FontName', fontName);
set(get(gca, 'title'), 'FontSize', fsize, 'FontName', fontName);
set(get(gca, 'XLabel'), 'FontSize', fsize, 'FontName', fontName);
set(get(gca, 'YLabel'), 'FontSize', fsize, 'FontName', fontName);
set(get(gca, 'ZLabel'), 'FontSize', fsize, 'FontName', fontName);
