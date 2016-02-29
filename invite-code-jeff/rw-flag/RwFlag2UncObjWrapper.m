function [ f, g ] = RwFlag2UncObjWrapper(x, V, D, c)

global RwFlag2UncObjWrapperParam
RwFlag2UncObjWrapperParam.D = D;
RwFlag2UncObjWrapperParam.c = c;
RwFlag2UncObjWrapperParam.mask = RwFlag2UncMask(V);
RwFlag2UncObjWrapperParam.V = V;

[f,g] = RwFlag2UncObj(x, 'RwFlag2UncObjWrapperParam');

end
