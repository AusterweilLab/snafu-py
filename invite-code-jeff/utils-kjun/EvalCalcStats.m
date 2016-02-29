function out= EvalCalcStats(col)
n = length(col);
sorted = sort(col,'ascend');

out = struct();
out.mean = mean(col);
out.dev95 = EvalCalcDeviation(col');
out.median = median(col);
out.min = min(col);
out.max = max(col);
out.top5perc = sorted( round(.05 * n));
out.bot5perc = sorted( round(.95 * n));
out.top2p5perc = sorted( round(.025 * n));
out.bot2p5perc = sorted( round(.975 * n));
end