<script setup lang="ts">
defineProps<{ label: string; value: string | number; hint?: string; tone?: 'blue' | 'red' | 'orange' | 'green' }>()
</script>

<template>
  <article class="metric-card" :class="`tone-${tone || 'blue'}`">
    <span>{{ label }}</span><strong>{{ value }}</strong><small v-if="hint">{{ hint }}</small>
  </article>
</template>
