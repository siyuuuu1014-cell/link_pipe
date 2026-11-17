import { defineStore } from 'pinia'
import { api } from '../api'

export const useSystemStore = defineStore('system', {
  state: () => ({
    health: null as Record<string, any> | null,
    metrics: null as Record<string, any> | null,
    loading: false,
  }),
  actions: {
    async refresh() {
      this.loading = true
      try {
        const [health, metrics] = await Promise.all([api.get('/health'), api.get('/metrics')])
        this.health = health.data
        this.metrics = metrics.data
      } finally {
        this.loading = false
      }
    },
  },
})
