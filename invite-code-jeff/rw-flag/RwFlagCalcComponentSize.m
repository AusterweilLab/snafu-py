function componentSize = RwFlagCalcComponentSize(P, s)

visited = false(size(P,1), 1);

visit(s);

componentSize = sum(visited);

  function visit(node)
    if (~visited(node))
      visited(node) = true;
      
      children = find(P(node,:) ~= 0);
      for child=children
        visit(child);
      end
    end
  end

end
