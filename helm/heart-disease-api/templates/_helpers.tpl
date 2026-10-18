{{- define "heart-disease-api.name" -}}
{{ .Chart.Name }}
{{- end -}}

{{- define "heart-disease-api.labels" -}}
app: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/name: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}
