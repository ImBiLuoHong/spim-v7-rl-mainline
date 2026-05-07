import torch
from typing import Dict, Any
from src.modeling.interfaces.base import NavigatorBase, NavigatorCapabilities
from src.modeling.registry import NAVIGATOR_REGISTRY, NAV_BACKBONE_REGISTRY, NAV_HEAD_REGISTRY

@NAVIGATOR_REGISTRY.register("composed_navigator")
class ComposedNavigator(NavigatorBase):
    """
    Standard implementation of NavigatorBase that composes a Backbone, a Head, and a Sampler.
    """
    def __init__(self, backbone, head, sampler, cfg=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.sampler = sampler
        self.cfg = cfg

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        # 1. Get node embeddings from backbone
        # Check if backbone supports memory (GRU)
        # Note: We need to pass batch vector if using global pooling
        memory_state = state.get('nav_memory_state')
        batch = getattr(graph, 'batch', None)
        
        # Adaptive call based on signature
        import inspect
        sig = inspect.signature(self.backbone.forward)
        
        # Prepare explicit context (Exp F)
        explicit_context = state.get('explicit_context')
        
        call_kwargs = {'memory_state': memory_state, 'batch': batch}
        if 'explicit_context' in sig.parameters:
            call_kwargs['explicit_context'] = explicit_context
            
        if 'memory_state' in sig.parameters:
            node_embeddings, next_memory_state = self.backbone(state['x_nav'], graph.edge_index, getattr(graph, 'edge_attr', None), **call_kwargs)
            # state['nav_memory_state'] = next_memory_state # Update state in-place for loop
        else:
            node_embeddings = self.backbone(state['x_nav'], graph.edge_index, getattr(graph, 'edge_attr', None))
            next_memory_state = None
        
        # 2. Add physics logit bias if present in state (passed from Orchestrator)
        if physics_ctx and 'logit_bias' in physics_ctx:
            # We pass physics_ctx to the head so it can decide how to use it
            state['_physics_ctx'] = physics_ctx
            
        # [Fix] Ensure batch is in state for Sampler
        if 'batch' not in state and hasattr(graph, 'batch'):
            state['batch'] = graph.batch
            
        # 3. Get logits from head
        logits = self.head(node_embeddings, state)
        
        # 4. Sampling
        outputs = self.sampler(logits, state)
        
        # 5. Standardize output keys for NavigatorBase
        # The Orchestrator expects 'logits' and optionally 'value'
        outputs['logits'] = logits
        
        if next_memory_state is not None:
            outputs['updated_memory_state'] = next_memory_state
        
        return outputs

    def capabilities(self) -> NavigatorCapabilities:
        # Merged capabilities from head and sampler
        caps = self.head.capabilities()
        caps.update(self.sampler.capabilities())
        return caps
